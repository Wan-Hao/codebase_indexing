use napi_derive::napi;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// A node in the Merkle tree
#[napi(object)]
#[derive(Clone, Debug)]
pub struct MerkleNode {
    /// Relative path of this node (file or directory)
    pub path: String,
    /// SHA-256 hash
    pub hash: String,
    /// Whether this is a file (leaf) or directory (internal node)
    pub is_file: bool,
    /// Children paths (empty for files)
    pub children: Vec<String>,
}

/// Result of diffing two Merkle trees
#[napi(object)]
#[derive(Clone, Debug)]
pub struct MerkleDiff {
    /// Files that were added (exist in new but not old)
    pub added: Vec<String>,
    /// Files that were removed (exist in old but not new)
    pub removed: Vec<String>,
    /// Files that were modified (exist in both but hash differs)
    pub modified: Vec<String>,
}

/// Build a Merkle tree from a list of (relative_path, file_content_hash) pairs.
/// Returns a list of all nodes (files + directories + root).
#[napi]
pub fn build_merkle_tree(file_hashes: Vec<FileHashEntry>) -> Vec<MerkleNode> {
    // Group files by directory
    let mut dir_children: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut nodes: BTreeMap<String, MerkleNode> = BTreeMap::new();

    // Insert all file (leaf) nodes
    for fh in &file_hashes {
        nodes.insert(
            fh.path.clone(),
            MerkleNode {
                path: fh.path.clone(),
                hash: fh.hash.clone(),
                is_file: true,
                children: vec![],
            },
        );

        // Register this file under its parent directory
        let parent = parent_path(&fh.path);
        dir_children
            .entry(parent.clone())
            .or_default()
            .push(fh.path.clone());

        // Ensure all ancestor directories are registered
        let mut current_parent = parent;
        loop {
            let grandparent = parent_path(&current_parent);
            if grandparent == current_parent {
                break; // reached root
            }
            dir_children
                .entry(grandparent.clone())
                .or_default();
            // Make sure current_parent is a child of grandparent
            let siblings = dir_children.entry(grandparent.clone()).or_default();
            if !siblings.contains(&current_parent) {
                siblings.push(current_parent.clone());
            }
            current_parent = grandparent;
        }
    }

    // Build directory nodes bottom-up (BTreeMap is sorted, process deepest paths first)
    // Collect all directory paths and sort by depth descending
    let mut dir_paths: Vec<String> = dir_children.keys().cloned().collect();
    dir_paths.sort_by(|a, b| {
        let depth_a = a.matches('/').count();
        let depth_b = b.matches('/').count();
        depth_b.cmp(&depth_a) // deepest first
    });

    for dir_path in &dir_paths {
        let children = dir_children.get(dir_path).cloned().unwrap_or_default();
        let mut child_hashes: Vec<String> = Vec::new();

        for child in &children {
            if let Some(node) = nodes.get(child) {
                child_hashes.push(node.hash.clone());
            }
        }

        // Sort child hashes for deterministic tree
        child_hashes.sort();
        let combined = child_hashes.join("");
        let mut hasher = Sha256::new();
        hasher.update(combined.as_bytes());
        let dir_hash = hex::encode(hasher.finalize());

        nodes.insert(
            dir_path.clone(),
            MerkleNode {
                path: dir_path.clone(),
                hash: dir_hash,
                is_file: false,
                children,
            },
        );
    }

    nodes.into_values().collect()
}

/// Diff two Merkle trees (represented as flat lists of nodes).
/// Returns added, removed, and modified FILE paths.
#[napi]
pub fn diff_merkle_trees(old_nodes: Vec<MerkleNode>, new_nodes: Vec<MerkleNode>) -> MerkleDiff {
    let old_files: BTreeMap<String, String> = old_nodes
        .iter()
        .filter(|n| n.is_file)
        .map(|n| (n.path.clone(), n.hash.clone()))
        .collect();

    let new_files: BTreeMap<String, String> = new_nodes
        .iter()
        .filter(|n| n.is_file)
        .map(|n| (n.path.clone(), n.hash.clone()))
        .collect();

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut modified = Vec::new();

    // Find added and modified
    for (path, new_hash) in &new_files {
        match old_files.get(path) {
            None => added.push(path.clone()),
            Some(old_hash) if old_hash != new_hash => modified.push(path.clone()),
            _ => {} // unchanged
        }
    }

    // Find removed
    for path in old_files.keys() {
        if !new_files.contains_key(path) {
            removed.push(path.clone());
        }
    }

    MerkleDiff {
        added,
        removed,
        modified,
    }
}

/// Get root hash from a list of Merkle nodes
#[napi]
pub fn get_root_hash(nodes: Vec<MerkleNode>) -> Option<String> {
    // Root is the node with the shortest path (or "." or "")
    nodes
        .iter()
        .min_by_key(|n| n.path.len())
        .map(|n| n.hash.clone())
}

#[napi(object)]
#[derive(Clone)]
pub struct FileHashEntry {
    pub path: String,
    pub hash: String,
}

fn parent_path(path: &str) -> String {
    match path.rfind('/') {
        Some(idx) => path[..idx].to_string(),
        None => ".".to_string(),
    }
}
