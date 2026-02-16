use napi_derive::napi;
use ignore::WalkBuilder;
use std::path::Path;

/// Scan a directory and return all indexable file paths.
/// Respects .gitignore and .cursorignore rules.
/// Filters to only include files with specified extensions.
#[napi]
pub fn scan_directory(root_path: String, extensions: Vec<String>) -> napi::Result<Vec<String>> {
    let root = Path::new(&root_path);
    if !root.is_dir() {
        return Err(napi::Error::from_reason(format!(
            "Not a directory: {}",
            root_path
        )));
    }

    let ext_set: std::collections::HashSet<String> = extensions
        .into_iter()
        .map(|e| e.trim_start_matches('.').to_lowercase())
        .collect();

    let mut files = Vec::new();

    let walker = WalkBuilder::new(&root_path)
        .hidden(true) // skip hidden files/dirs
        .git_ignore(true) // respect .gitignore
        .git_global(true)
        .git_exclude(true)
        .add_custom_ignore_filename(".cursorignore")
        .build();

    for entry in walker {
        let entry = entry.map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        // Filter by extension
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            if ext_set.is_empty() || ext_set.contains(&ext_str) {
                if let Some(path_str) = path.to_str() {
                    files.push(path_str.to_string());
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

/// Get relative path from root
#[napi]
pub fn get_relative_path(root_path: String, file_path: String) -> Option<String> {
    let root = Path::new(&root_path);
    let file = Path::new(&file_path);
    file.strip_prefix(root)
        .ok()
        .and_then(|p| p.to_str())
        .map(|s| s.to_string())
}
