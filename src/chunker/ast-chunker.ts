/**
 * AST-based code chunker using tree-sitter.
 *
 * Strategy:
 * 1. Parse file into AST using tree-sitter
 * 2. Extract top-level semantic nodes (functions, classes, interfaces, etc.)
 * 3. Merge small sibling nodes to meet minimum chunk size
 * 4. Split oversized nodes (methods within classes) if they exceed max chunk size
 * 5. Attach metadata (file path, line range, node type, symbol name)
 */

import Parser from 'tree-sitter';
import TypeScript from 'tree-sitter-typescript';
import crypto from 'node:crypto';
import type { CodeChunk } from '../types.js';

// Top-level AST node types that represent meaningful semantic boundaries
const SEMANTIC_NODE_TYPES = new Set([
  // Declarations
  'function_declaration',
  'generator_function_declaration',
  'class_declaration',
  'abstract_class_declaration',
  'interface_declaration',
  'type_alias_declaration',
  'enum_declaration',
  'module', // namespace
  // Statements
  'export_statement',
  'import_statement',
  'lexical_declaration', // const/let/var at top level
  'variable_declaration',
  'expression_statement',
]);

// Node types that can contain extractable children (e.g., class body)
const CONTAINER_NODE_TYPES = new Set([
  'class_declaration',
  'abstract_class_declaration',
  'interface_declaration',
]);

// Method-level nodes within containers
const METHOD_NODE_TYPES = new Set([
  'method_definition',
  'public_field_definition',
  'property_signature',
  'method_signature',
  'abstract_method_signature',
  'property_definition',
]);

export class ASTChunker {
  private parser: Parser;
  private maxChunkTokens: number;
  private minChunkTokens: number;

  constructor(maxChunkTokens = 512, minChunkTokens = 30) {
    this.parser = new Parser();
    this.parser.setLanguage(TypeScript.typescript);
    this.maxChunkTokens = maxChunkTokens;
    this.minChunkTokens = minChunkTokens;
  }

  /**
   * Set language to TypeScript TSX (for .tsx files)
   */
  setTSX(): void {
    this.parser.setLanguage(TypeScript.tsx);
  }

  /**
   * Set language to regular TypeScript (for .ts files)
   */
  setTS(): void {
    this.parser.setLanguage(TypeScript.typescript);
  }

  /**
   * Chunk a file's content into semantically meaningful code blocks.
   */
  chunkFile(filePath: string, content: string): CodeChunk[] {
    // Choose parser based on extension
    if (filePath.endsWith('.tsx')) {
      this.setTSX();
    } else {
      this.setTS();
    }

    const tree = this.parser.parse(content);
    const rootNode = tree.rootNode;
    const lines = content.split('\n');

    // Step 1: Extract top-level semantic nodes
    const rawSegments = this.extractTopLevelSegments(rootNode);

    // Step 2: Handle oversized container nodes (split class methods)
    const expandedSegments = this.expandLargeContainers(rawSegments, lines);

    // Step 3: Merge small adjacent segments
    const mergedSegments = this.mergeSmallSegments(expandedSegments, lines);

    // Step 4: Convert segments to CodeChunks
    const chunks: CodeChunk[] = mergedSegments.map((seg) => {
      const chunkContent = lines.slice(seg.startLine, seg.endLine + 1).join('\n');
      const contentHash = crypto.createHash('sha256').update(chunkContent).digest('hex');

      return {
        chunkId: contentHash,
        filePath,
        startLine: seg.startLine + 1, // convert to 1-based
        endLine: seg.endLine + 1,
        content: chunkContent,
        contentHash,
        nodeType: seg.nodeType,
        symbolName: seg.symbolName,
      };
    });

    return chunks;
  }

  /**
   * Extract top-level children of the root AST node as segments.
   */
  private extractTopLevelSegments(rootNode: Parser.SyntaxNode): Segment[] {
    const segments: Segment[] = [];

    for (const child of rootNode.children) {
      if (child.type === 'comment' || child.type === 'ERROR') {
        continue;
      }

      const segment: Segment = {
        startLine: child.startPosition.row,
        endLine: child.endPosition.row,
        nodeType: child.type,
        symbolName: this.extractSymbolName(child),
        children: [],
      };

      // If this is a container (class/interface), extract its children too
      if (CONTAINER_NODE_TYPES.has(child.type)) {
        segment.children = this.extractContainerChildren(child);
      }

      segments.push(segment);
    }

    return segments;
  }

  /**
   * Extract method-level children from a container node (class/interface).
   */
  private extractContainerChildren(node: Parser.SyntaxNode): Segment[] {
    const children: Segment[] = [];
    const body = node.childForFieldName('body');
    if (!body) return children;

    for (const child of body.children) {
      if (METHOD_NODE_TYPES.has(child.type)) {
        children.push({
          startLine: child.startPosition.row,
          endLine: child.endPosition.row,
          nodeType: child.type,
          symbolName: this.extractSymbolName(child),
          children: [],
        });
      }
    }

    return children;
  }

  /**
   * If a container node is too large, split it into header + individual methods.
   */
  private expandLargeContainers(segments: Segment[], lines: string[]): Segment[] {
    const result: Segment[] = [];

    for (const seg of segments) {
      const tokenCount = this.estimateTokens(lines, seg.startLine, seg.endLine);

      if (tokenCount > this.maxChunkTokens && seg.children.length > 0) {
        // Split: emit each method as a separate segment
        // Include the class/interface header with the first method
        let lastEnd = seg.startLine;

        for (const child of seg.children) {
          // Gap between last end and this child's start (includes decorators, comments)
          if (child.startLine > lastEnd) {
            const gapTokens = this.estimateTokens(lines, lastEnd, child.startLine - 1);
            if (gapTokens >= this.minChunkTokens) {
              result.push({
                startLine: lastEnd,
                endLine: child.startLine - 1,
                nodeType: seg.nodeType + '_header',
                symbolName: seg.symbolName,
                children: [],
              });
            }
          }

          result.push({
            startLine: Math.min(lastEnd, child.startLine),
            endLine: child.endLine,
            nodeType: child.nodeType,
            symbolName: child.symbolName
              ? `${seg.symbolName}.${child.symbolName}`
              : seg.symbolName,
            children: [],
          });

          lastEnd = child.endLine + 1;
        }

        // Trailing content (closing brace, etc.)
        if (lastEnd <= seg.endLine) {
          const trailingTokens = this.estimateTokens(lines, lastEnd, seg.endLine);
          if (trailingTokens >= this.minChunkTokens) {
            result.push({
              startLine: lastEnd,
              endLine: seg.endLine,
              nodeType: seg.nodeType + '_footer',
              symbolName: seg.symbolName,
              children: [],
            });
          } else if (result.length > 0) {
            // Merge trailing into last segment
            result[result.length - 1].endLine = seg.endLine;
          }
        }
      } else {
        // Keep as-is (small enough or no children to split)
        result.push({
          startLine: seg.startLine,
          endLine: seg.endLine,
          nodeType: seg.nodeType,
          symbolName: seg.symbolName,
          children: [],
        });
      }
    }

    return result;
  }

  /**
   * Merge adjacent small segments until they meet the minimum token threshold.
   */
  private mergeSmallSegments(segments: Segment[], lines: string[]): Segment[] {
    if (segments.length === 0) return [];

    const result: Segment[] = [];
    let current = { ...segments[0] };

    for (let i = 1; i < segments.length; i++) {
      const currentTokens = this.estimateTokens(lines, current.startLine, current.endLine);
      const nextTokens = this.estimateTokens(
        lines,
        segments[i].startLine,
        segments[i].endLine
      );

      if (currentTokens < this.minChunkTokens || nextTokens < this.minChunkTokens) {
        // Merge: extend current to include next
        current.endLine = segments[i].endLine;
        // Keep the more descriptive node type
        if (currentTokens < nextTokens) {
          current.nodeType = segments[i].nodeType;
          current.symbolName = segments[i].symbolName;
        }
      } else {
        result.push(current);
        current = { ...segments[i] };
      }
    }

    result.push(current);
    return result;
  }

  /**
   * Extract symbol name from an AST node.
   */
  private extractSymbolName(node: Parser.SyntaxNode): string | undefined {
    // Try 'name' field first
    const nameNode = node.childForFieldName('name');
    if (nameNode) {
      return nameNode.text;
    }

    // For export statements, look at the declaration inside
    if (node.type === 'export_statement') {
      const decl = node.childForFieldName('declaration');
      if (decl) {
        return this.extractSymbolName(decl);
      }
      // export default
      const value = node.childForFieldName('value');
      if (value) {
        return 'default';
      }
    }

    return undefined;
  }

  /**
   * Rough token estimation: ~4 characters per token (GPT-like tokenizer heuristic).
   */
  private estimateTokens(lines: string[], startLine: number, endLine: number): number {
    let charCount = 0;
    for (let i = startLine; i <= Math.min(endLine, lines.length - 1); i++) {
      charCount += lines[i].length + 1; // +1 for newline
    }
    return Math.ceil(charCount / 4);
  }
}

/** Internal representation of a code segment before it becomes a CodeChunk */
interface Segment {
  startLine: number; // 0-based
  endLine: number; // 0-based
  nodeType: string;
  symbolName?: string;
  children: Segment[];
}
