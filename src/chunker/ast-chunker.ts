/**
 * AST-based code chunker using tree-sitter.
 *
 * Strategy:
 * 1. Parse file into AST using tree-sitter
 * 2. Extract top-level semantic nodes (functions, classes, interfaces, etc.)
 *    — Attach preceding comments (JSDoc, etc.) to their corresponding node
 * 3. Split oversized containers (class → methods) and functions (→ logical blocks)
 * 4. Merge small sibling nodes to meet minimum chunk size
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

// Function-like node types that can be split by internal logic
const FUNCTION_NODE_TYPES = new Set([
  'function_declaration',
  'generator_function_declaration',
  'arrow_function',
  'function',
  'method_definition',
]);

// Property types inside object literals, used for splitting large config objects
const OBJECT_PROPERTY_TYPES = new Set([
  'pair',                // key: value
  'method_definition',   // shorthand methods: execute() { ... }
  'spread_element',      // ...spread
  'shorthand_property_identifier',
]);

// Logical block types inside function bodies, used for splitting large functions
const LOGICAL_BLOCK_TYPES = new Set([
  'if_statement',
  'for_statement',
  'for_in_statement',
  'while_statement',
  'do_statement',
  'switch_statement',
  'try_statement',
  'return_statement',
  'throw_statement',
  'lexical_declaration',
  'variable_declaration',
  'expression_statement',
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

    // Step 1: Extract top-level semantic nodes (with comments attached)
    const rawSegments = this.extractTopLevelSegments(rootNode);

    // Step 2: Split oversized nodes (class → methods, function → logical blocks)
    const expandedSegments = this.expandOversizedSegments(rawSegments, lines);

    // Step 3: Merge small adjacent segments
    const mergedSegments = this.mergeSmallSegments(expandedSegments, lines);

    // Step 4: Convert segments to CodeChunks
    const chunks: CodeChunk[] = mergedSegments.map((seg) => {
      const chunkContent = lines.slice(seg.startLine, seg.endLine + 1).join('\n');
      const contentHash = crypto
        .createHash('sha256')
        .update(chunkContent)
        .digest('hex');

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

  // ---------------------------------------------------------------------------
  // Step 1: Extraction
  // ---------------------------------------------------------------------------

  /**
   * Extract top-level children of the root AST node as segments.
   * Preceding comments are attached to the next semantic node.
   * (Fix #1: comment attachment, Fix #4: SEMANTIC_NODE_TYPES filter)
   */
  private extractTopLevelSegments(rootNode: Parser.SyntaxNode): Segment[] {
    const segments: Segment[] = [];
    let pendingCommentStart: number | null = null;

    for (const child of rootNode.children) {
      // Track consecutive comment nodes — will be attached to the next semantic node
      if (child.type === 'comment') {
        if (pendingCommentStart === null) {
          pendingCommentStart = child.startPosition.row;
        }
        continue;
      }

      // Skip error nodes and non-semantic nodes (Fix #4)
      if (child.type === 'ERROR' || !SEMANTIC_NODE_TYPES.has(child.type)) {
        pendingCommentStart = null; // reset — don't attach orphan comments
        continue;
      }

      // Start line absorbs any preceding comments (Fix #1)
      const startLine =
        pendingCommentStart !== null
          ? pendingCommentStart
          : child.startPosition.row;
      pendingCommentStart = null;

      const segment: Segment = {
        startLine,
        endLine: child.endPosition.row,
        nodeType: child.type,
        symbolName: this.extractSymbolName(child),
        children: [],
      };

      // If this is (or wraps) a container (class/interface), extract method children
      const containerNode = this.findContainerNode(child);
      if (containerNode) {
        segment.children = this.extractContainerChildren(containerNode);
      }

      // If this wraps a function, extract logical blocks for potential splitting
      const funcNode = !containerNode ? this.findFunctionNode(child) : null;
      if (funcNode) {
        segment.functionBlocks = this.extractFunctionBlocks(funcNode);
      }

      // If this wraps an object literal (and is not already a container or function),
      // extract its properties for potential splitting
      if (segment.children.length === 0 && !funcNode) {
        const objNode = this.findObjectLiteral(child);
        if (objNode) {
          segment.children = this.extractObjectProperties(objNode);
        }
      }

      segments.push(segment);
    }

    return segments;
  }

  /**
   * Extract method-level children from a container node (class/interface).
   * Preceding comments within the class body are attached to the next method.
   * (Fix #1: comment attachment for methods)
   */
  private extractContainerChildren(node: Parser.SyntaxNode): Segment[] {
    const children: Segment[] = [];
    const body = node.childForFieldName('body');
    if (!body) return children;

    let pendingCommentStart: number | null = null;

    for (const child of body.children) {
      if (child.type === 'comment') {
        if (pendingCommentStart === null) {
          pendingCommentStart = child.startPosition.row;
        }
        continue;
      }

      if (METHOD_NODE_TYPES.has(child.type)) {
        const startLine =
          pendingCommentStart !== null
            ? pendingCommentStart
            : child.startPosition.row;
        pendingCommentStart = null;

        children.push({
          startLine,
          endLine: child.endPosition.row,
          nodeType: child.type,
          symbolName: this.extractSymbolName(child),
          children: [],
        });
      } else {
        // Non-method, non-comment node (e.g., '{', '}', decorator) — reset pending
        pendingCommentStart = null;
      }
    }

    return children;
  }

  /**
   * Extract logical block children from a function body for sub-function splitting.
   * Handles "factory" patterns where the outer function body is a single return
   * of an arrow/anonymous function — recurses into the inner function's body.
   * (Fix #2: function-level splitting support)
   */
  private extractFunctionBlocks(node: Parser.SyntaxNode): Segment[] {
    const body = node.childForFieldName('body');
    if (!body || body.type !== 'statement_block') return [];

    // Collect non-brace children
    const stmts = body.children.filter(
      (c) => c.type !== '{' && c.type !== '}'
    );

    // Factory pattern: function body is a single return of an arrow/anonymous function
    // e.g. function create() { return async (api) => { ...actual logic... }; }
    if (stmts.length === 1 && stmts[0].type === 'return_statement') {
      const innerFunc = this.findInnerFunction(stmts[0]);
      if (innerFunc) {
        return this.extractFunctionBlocks(innerFunc);
      }
    }

    return this.collectLogicalBlocks(stmts);
  }

  /**
   * Collect logical blocks from a list of statements.
   * If a block is large (>15 lines) and has inner sub-statements,
   * recursively expand it into finer-grained split points.
   * This handles nested if/for/try blocks that would otherwise
   * produce oversized chunks.
   */
  private collectLogicalBlocks(
    stmts: Parser.SyntaxNode[],
    maxDepth = 2
  ): Segment[] {
    const blocks: Segment[] = [];

    for (const child of stmts) {
      if (!LOGICAL_BLOCK_TYPES.has(child.type)) continue;

      const blockLines =
        child.endPosition.row - child.startPosition.row + 1;

      // If block is large and we haven't hit max depth, try to expand
      if (blockLines > 15 && maxDepth > 0) {
        const innerStmts = this.getBodyStatements(child);
        if (innerStmts.length > 1) {
          const innerBlocks = this.collectLogicalBlocks(
            innerStmts,
            maxDepth - 1
          );
          if (innerBlocks.length > 1) {
            blocks.push(...innerBlocks);
            continue;
          }
        }
      }

      // Keep as a single block
        blocks.push({
          startLine: child.startPosition.row,
          endLine: child.endPosition.row,
          nodeType: child.type,
          symbolName: undefined,
          children: [],
        });
    }

    return blocks;
  }

  /**
   * Extract direct child statements from all statement_block bodies
   * of a compound statement (if body, else body, catch body, etc).
   * Also handles else-if chains by treating the inner if_statement
   * as a block candidate for further expansion.
   */
  private getBodyStatements(node: Parser.SyntaxNode): Parser.SyntaxNode[] {
    const stmts: Parser.SyntaxNode[] = [];

    for (const child of node.children) {
      // Direct body: if consequence, for body, try body, while body
      if (child.type === 'statement_block') {
        for (const stmt of child.children) {
          if (stmt.type !== '{' && stmt.type !== '}') {
            stmts.push(stmt);
          }
        }
      }
      // else_clause, catch_clause, finally_clause wrap statement_blocks
      if (
        child.type === 'else_clause' ||
        child.type === 'catch_clause' ||
        child.type === 'finally_clause'
      ) {
        for (const inner of child.children) {
          if (inner.type === 'statement_block') {
            for (const stmt of inner.children) {
              if (stmt.type !== '{' && stmt.type !== '}') {
                stmts.push(stmt);
              }
            }
          }
          // else-if chain: treat inner if_statement as a block
          if (inner.type === 'if_statement') {
            stmts.push(inner);
          }
        }
      }
    }

    return stmts;
  }

  /**
   * Find an arrow_function or function expression nested inside a return statement.
   * Traverses through parenthesized expressions, await expressions, etc.
   */
  private findInnerFunction(
    node: Parser.SyntaxNode
  ): Parser.SyntaxNode | null {
    for (const child of node.children) {
      if (FUNCTION_NODE_TYPES.has(child.type)) {
        return child;
      }
      // Traverse through parenthesized_expression, await_expression, etc.
      if (
        child.type === 'parenthesized_expression' ||
        child.type === 'await_expression' ||
        child.type === 'as_expression'
      ) {
        const found = this.findInnerFunction(child);
        if (found) return found;
      }
    }
    return null;
  }

  /**
   * Find a container (class/interface) inside an export_statement or directly.
   * e.g. `export class Foo { ... }` → the class_declaration
   * e.g. `export abstract class Foo { ... }` → the abstract_class_declaration
   */
  private findContainerNode(
    node: Parser.SyntaxNode
  ): Parser.SyntaxNode | null {
    if (CONTAINER_NODE_TYPES.has(node.type)) {
      return node;
    }

    // export statement wrapping a container
    if (node.type === 'export_statement') {
      const decl = node.childForFieldName('declaration');
      if (decl && CONTAINER_NODE_TYPES.has(decl.type)) {
        return decl;
      }
    }

    return null;
  }

  /**
   * Find the actual function node inside an export_statement or variable declaration.
   * e.g. `export function foo()` → the function_declaration
   * e.g. `const foo = () => {}` → the arrow_function
   */
  private findFunctionNode(
    node: Parser.SyntaxNode
  ): Parser.SyntaxNode | null {
    if (FUNCTION_NODE_TYPES.has(node.type)) {
      return node;
    }

    // export statement wrapping a function
    if (node.type === 'export_statement') {
      const decl = node.childForFieldName('declaration');
      if (decl && FUNCTION_NODE_TYPES.has(decl.type)) {
        return decl;
      }
      const value = node.childForFieldName('value');
      if (value && FUNCTION_NODE_TYPES.has(value.type)) {
        return value;
      }
    }

    // const foo = () => {} or const foo = function() {}
    if (
      node.type === 'lexical_declaration' ||
      node.type === 'variable_declaration'
    ) {
      for (const child of node.children) {
        if (child.type === 'variable_declarator') {
          const value = child.childForFieldName('value');
          if (value && FUNCTION_NODE_TYPES.has(value.type)) {
            return value;
          }
        }
      }
    }

    return null;
  }

  /**
   * Find an object literal inside an export_statement or variable declaration.
   * e.g. `export const foo: Type = { ... }` → the object node
   * Also handles `satisfies` / `as` wrappers.
   */
  private findObjectLiteral(
    node: Parser.SyntaxNode
  ): Parser.SyntaxNode | null {
    // Unwrap export_statement
    if (node.type === 'export_statement') {
      const decl = node.childForFieldName('declaration');
      if (decl) return this.findObjectLiteral(decl);
      const value = node.childForFieldName('value');
      if (value) return this.findObjectLiteral(value);
      return null;
    }

    // Unwrap lexical_declaration / variable_declaration
    if (
      node.type === 'lexical_declaration' ||
      node.type === 'variable_declaration'
    ) {
      for (const child of node.children) {
        if (child.type === 'variable_declarator') {
          const value = child.childForFieldName('value');
          if (value) return this.findObjectLiteral(value);
        }
      }
      return null;
    }

    // Unwrap `satisfies Type` / `as Type` wrappers
    if (
      node.type === 'satisfies_expression' ||
      node.type === 'as_expression'
    ) {
      // The object is the first child (left operand)
      for (const child of node.children) {
        if (child.type === 'object') return child;
      }
      return null;
    }

    // Direct object literal
    if (node.type === 'object') return node;

    return null;
  }

  /**
   * Extract top-level properties from an object literal.
   * Preceding comments are attached to the next property.
   */
  private extractObjectProperties(
    objectNode: Parser.SyntaxNode
  ): Segment[] {
    const properties: Segment[] = [];
    let pendingCommentStart: number | null = null;

    for (const child of objectNode.children) {
      if (child.type === 'comment') {
        if (pendingCommentStart === null) {
          pendingCommentStart = child.startPosition.row;
        }
        continue;
      }

      if (OBJECT_PROPERTY_TYPES.has(child.type)) {
        const startLine =
          pendingCommentStart !== null
            ? pendingCommentStart
            : child.startPosition.row;
        pendingCommentStart = null;

        const prop: Segment = {
          startLine,
          endLine: child.endPosition.row,
          nodeType: child.type,
          symbolName: this.extractPropertyName(child),
          children: [],
        };

        // For method_definition inside objects, extract function blocks
        // so they can be further split if oversized
        if (child.type === 'method_definition') {
          prop.functionBlocks = this.extractFunctionBlocks(child);
        }

        properties.push(prop);
      } else {
        // Braces, commas, etc. — reset pending comment
        pendingCommentStart = null;
      }
    }

    return properties;
  }

  /**
   * Extract property name from an object property node (pair or method_definition).
   */
  private extractPropertyName(node: Parser.SyntaxNode): string | undefined {
    // pair: key is the first named child or 'key' field
    if (node.type === 'pair') {
      const key = node.childForFieldName('key');
      if (key) return key.text;
    }
    // method_definition: has a 'name' field
    if (node.type === 'method_definition') {
      const name = node.childForFieldName('name');
      if (name) return name.text;
    }
    // spread_element: use the text minus the dots
    if (node.type === 'spread_element') {
      return '...' + (node.children[1]?.text ?? '');
    }
    return undefined;
  }

  // ---------------------------------------------------------------------------
  // Step 2: Expansion (split oversized segments)
  // ---------------------------------------------------------------------------

  /**
   * Split oversized segments: containers by methods/properties, functions by
   * logical blocks. Recursive: sub-results from container splitting are checked
   * again (e.g., a method_definition extracted from an object literal may itself
   * need function-level splitting).
   */
  private expandOversizedSegments(
    segments: Segment[],
    lines: string[]
  ): Segment[] {
    const result: Segment[] = [];

    for (const seg of segments) {
      this.expandSegment(seg, lines, result);
    }

    return result;
  }

  /**
   * Recursively expand a single segment if it exceeds maxChunkTokens.
   */
  private expandSegment(
    seg: Segment,
    lines: string[],
    result: Segment[]
  ): void {
    // Container (class/interface/object) with children → split then recurse
      if (seg.children.length > 0) {
      const tokenCount = this.estimateTokens(
        lines,
        seg.startLine,
        seg.endLine
      );
        if (tokenCount > this.maxChunkTokens) {
        const subResults: Segment[] = [];
        this.splitContainer(seg, lines, subResults);
        // Recursively expand each sub-result (e.g., method inside object)
        for (const sub of subResults) {
          this.expandSegment(sub, lines, result);
        }
        return;
      }
    }

    // Function with logical blocks → split by blocks (leaf-level, no recursion)
    if (seg.functionBlocks && seg.functionBlocks.length > 1) {
      const tokenCount = this.estimateTokens(
        lines,
        seg.startLine,
        seg.endLine
      );
        if (tokenCount > this.maxChunkTokens) {
          this.splitByBlocks(seg, seg.functionBlocks, lines, result);
        return;
      }
      }

    // Safety net: if the segment is STILL oversized and couldn't be split by
    // any semantic strategy, fall back to line-based splitting.
    const finalTokens = this.estimateTokens(lines, seg.startLine, seg.endLine);
    if (finalTokens > this.maxChunkTokens) {
      this.splitByLineCount(seg, lines, result);
      return;
    }

    // Keep as-is (small enough)
    result.push({
      startLine: seg.startLine,
      endLine: seg.endLine,
      nodeType: seg.nodeType,
      symbolName: seg.symbolName,
      children: [],
    });
  }

  /**
   * Last-resort splitting: accumulate lines until the token budget is reached,
   * then start a new chunk. This handles uneven line lengths correctly.
   */
  private splitByLineCount(
    seg: Segment,
    lines: string[],
    result: Segment[]
  ): void {
    const maxChars = this.maxChunkTokens * 4; // inverse of estimateTokens
    let partIdx = 0;
    let chunkStart = seg.startLine;
    let chunkChars = 0;

    for (let i = seg.startLine; i <= seg.endLine; i++) {
      const lineChars = (lines[i]?.length ?? 0) + 1; // +1 for newline

      // If adding this line would exceed the budget AND we already have content
      if (chunkChars + lineChars > maxChars && i > chunkStart) {
        partIdx++;
        result.push({
          startLine: chunkStart,
          endLine: i - 1,
          nodeType: seg.nodeType + '_part',
          symbolName: seg.symbolName
            ? `${seg.symbolName} (part ${partIdx})`
            : `(part ${partIdx})`,
          children: [],
        });
        chunkStart = i;
        chunkChars = 0;
      }
      chunkChars += lineChars;
    }

    // Flush remaining lines
    if (chunkStart <= seg.endLine) {
      partIdx++;
      result.push({
        startLine: chunkStart,
        endLine: seg.endLine,
        nodeType: seg.nodeType + '_part',
        symbolName: seg.symbolName
          ? `${seg.symbolName} (part ${partIdx})`
          : `(part ${partIdx})`,
        children: [],
      });
    }
  }

  /**
   * Split a container (class/interface) into header + individual method segments.
   * (Fix #3: no overlapping ranges between header and method segments)
   */
  private splitContainer(
    seg: Segment,
    lines: string[],
    result: Segment[]
  ): void {
    let lastEnd = seg.startLine;

    for (const child of seg.children) {
      if (child.startLine > lastEnd) {
        const gapTokens = this.estimateTokens(
          lines,
          lastEnd,
          child.startLine - 1
        );

        const childSymbol = child.symbolName
          ? `${seg.symbolName}.${child.symbolName}`
          : seg.symbolName;

        if (gapTokens >= this.minChunkTokens) {
          // Gap is large enough → push as a separate header/gap segment
          result.push({
            startLine: lastEnd,
            endLine: child.startLine - 1,
            nodeType: seg.nodeType + '_header',
            symbolName: seg.symbolName,
            children: [],
          });
          // Method starts at its OWN line — no overlap (Fix #3)
          result.push({
            startLine: child.startLine,
            endLine: child.endLine,
            nodeType: child.nodeType,
            symbolName: childSymbol,
            children: [],
            functionBlocks: child.functionBlocks, // carry over for recursive expansion
          });
        } else {
          // Gap too small → absorb gap into method segment
          result.push({
            startLine: lastEnd,
            endLine: child.endLine,
            nodeType: child.nodeType,
            symbolName: childSymbol,
            children: [],
            functionBlocks: child.functionBlocks,
          });
        }
      } else {
        // No gap between previous child end and this child start
        const childSymbol = child.symbolName
          ? `${seg.symbolName}.${child.symbolName}`
          : seg.symbolName;

        result.push({
          startLine: Math.min(lastEnd, child.startLine),
          endLine: child.endLine,
          nodeType: child.nodeType,
          symbolName: childSymbol,
          children: [],
          functionBlocks: child.functionBlocks,
        });
      }

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
  }

  /**
   * Split a large function by its internal logical blocks.
   * Groups consecutive blocks to stay within maxChunkTokens.
   * (Fix #2: function-level splitting)
   */
  private splitByBlocks(
    seg: Segment,
    blocks: Segment[],
    lines: string[],
    result: Segment[]
  ): void {
    if (blocks.length === 0) {
      // Fallback: keep entire segment
      result.push({
        startLine: seg.startLine,
        endLine: seg.endLine,
        nodeType: seg.nodeType,
        symbolName: seg.symbolName,
        children: [],
      });
      return;
    }

    // Function signature (everything before the first logical block) as header
    const firstBlockStart = blocks[0].startLine;
    const headerEnd = firstBlockStart - 1;
    let headerPushed = false;

    if (firstBlockStart > seg.startLine) {
      const headerTokens = this.estimateTokens(
        lines,
        seg.startLine,
        headerEnd
      );
      if (headerTokens >= this.minChunkTokens) {
        result.push({
          startLine: seg.startLine,
          endLine: headerEnd,
          nodeType: seg.nodeType + '_header',
          symbolName: seg.symbolName,
          children: [],
        });
        headerPushed = true;
      }
    }

    // Group consecutive blocks so each group fits within maxChunkTokens
    // If header was too small, absorb it into the first block group
    let groupStart = headerPushed ? blocks[0].startLine : seg.startLine;
    let groupEnd = blocks[0].endLine;

    for (let i = 1; i < blocks.length; i++) {
      const extendedTokens = this.estimateTokens(
        lines,
        groupStart,
        blocks[i].endLine
      );

      if (extendedTokens > this.maxChunkTokens) {
        // Flush current group
        result.push({
          startLine: groupStart,
          endLine: groupEnd,
          nodeType: seg.nodeType + '_block',
          symbolName: seg.symbolName,
          children: [],
        });
        // Start new group right after the flushed chunk to avoid losing
        // lines between blocks (blank lines, comments, else/catch keywords)
        groupStart = groupEnd + 1;
        groupEnd = blocks[i].endLine;
      } else {
        groupEnd = blocks[i].endLine;
      }
    }

    // Flush last group — extend to seg.endLine to include closing brace
    result.push({
      startLine: groupStart,
      endLine: seg.endLine,
      nodeType: seg.nodeType + '_block',
      symbolName: seg.symbolName,
      children: [],
    });
  }

  // ---------------------------------------------------------------------------
  // Step 3: Merge small segments
  // ---------------------------------------------------------------------------

  /**
   * Merge adjacent small segments until they meet the minimum token threshold.
   * (Fix #6: cache token counts to avoid redundant computation)
   */
  private mergeSmallSegments(
    segments: Segment[],
    lines: string[]
  ): Segment[] {
    if (segments.length === 0) return [];

    const result: Segment[] = [];
    let current = { ...segments[0] };
    let currentTokens = this.estimateTokens(
      lines,
      current.startLine,
      current.endLine
    );

    for (let i = 1; i < segments.length; i++) {
      const nextTokens = this.estimateTokens(
        lines,
        segments[i].startLine,
        segments[i].endLine
      );

      if (
        currentTokens < this.minChunkTokens ||
        nextTokens < this.minChunkTokens
      ) {
        // Merge: extend current to include next
        current.endLine = segments[i].endLine;
        // Keep the more descriptive node type
        if (currentTokens < nextTokens) {
          current.nodeType = segments[i].nodeType;
          current.symbolName = segments[i].symbolName;
        }
        // Recompute token count for the merged range
        currentTokens = this.estimateTokens(
          lines,
          current.startLine,
          current.endLine
        );
      } else {
        result.push(current);
        current = { ...segments[i] };
        currentTokens = nextTokens; // reuse already-computed value (Fix #6)
      }
    }

    result.push(current);
    return result;
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

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
  private estimateTokens(
    lines: string[],
    startLine: number,
    endLine: number
  ): number {
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
  /** Logical blocks within a function body (for function-level splitting) */
  functionBlocks?: Segment[];
}
