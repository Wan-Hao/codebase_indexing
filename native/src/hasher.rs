use napi_derive::napi;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// Compute SHA-256 hash of a string
#[napi]
pub fn sha256_hash(content: String) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

/// Compute SHA-256 hash of a file's contents
#[napi]
pub fn sha256_hash_file(file_path: String) -> napi::Result<String> {
    let content = fs::read(&file_path).map_err(|e| {
        napi::Error::from_reason(format!("Failed to read file {}: {}", file_path, e))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    Ok(hex::encode(hasher.finalize()))
}

/// Batch compute SHA-256 hashes for multiple files (parallel via rayon)
#[napi]
pub fn sha256_hash_files(file_paths: Vec<String>) -> Vec<FileHash> {
    use rayon::prelude::*;

    file_paths
        .par_iter()
        .filter_map(|path| {
            let content = fs::read(path).ok()?;
            let mut hasher = Sha256::new();
            hasher.update(&content);
            Some(FileHash {
                path: path.clone(),
                hash: hex::encode(hasher.finalize()),
            })
        })
        .collect()
}

#[napi(object)]
#[derive(Clone)]
pub struct FileHash {
    pub path: String,
    pub hash: String,
}
