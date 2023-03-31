use anyhow::Error;
use atomicwrites::{AtomicFile, OverwriteBehavior};
use std::collections::HashSet;
use std::fs;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use terra_types::VNode;

lazy_static! {
    static ref TERRA_DIRECTORY: PathBuf =
        dirs::cache_dir().unwrap_or(PathBuf::from(".")).join("terra");
}

pub(crate) struct MapFile {
    server: String,
    remote_tiles: Arc<Mutex<HashSet<VNode>>>,
}
impl MapFile {
    pub(crate) async fn new(server: String) -> Result<Self, Error> {
        // Download file list if necessary.
        let file_list_path = TERRA_DIRECTORY.join("tile_list.txt.zstd");
        let file_list_encoded = if !file_list_path.exists() {
            let contents = Self::download(&server, "tile_list.txt.zstd").await?;
            if server.starts_with("http://") || server.starts_with("https://") {
                tokio::fs::write(&file_list_path, &contents).await?;
            }
            contents
        } else {
            tokio::fs::read(file_list_path).await?
        };

        // Parse file list to learn all files available from the remote.
        let remote_files = String::from_utf8(zstd::decode_all(Cursor::new(&file_list_encoded))?)?;
        let remote_tiles = remote_files
            .split('\n')
            .filter_map(|f| f.strip_suffix(".zip"))
            .map(VNode::from_str)
            .collect::<Result<HashSet<VNode>, Error>>()?;

        Ok(Self { server, remote_tiles: Arc::new(Mutex::new(remote_tiles)) })
    }

    pub(crate) async fn read_tile(&self, node: VNode) -> Result<Option<Vec<u8>>, Error> {
        let filename = TERRA_DIRECTORY.join("tiles").join(&format!("{}.zip", node));
        if filename.exists() {
            Ok(Some(tokio::fs::read(&filename).await?))
        } else {
            if !self.remote_tiles.lock().unwrap().contains(&node) {
                return Ok(None);
            }
            let contents = Self::download(&self.server, &format!("tiles/{}.zip", node)).await?;
            if self.server.starts_with("http://") || self.server.starts_with("https://") {
                if let Some(parent) = filename.parent() {
                    fs::create_dir_all(parent)?;
                }
                AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                    .write(|f| f.write_all(&contents))?;
            }
            Ok(Some(contents))
        }
    }

    pub(crate) async fn read_asset(&self, name: &str) -> Result<Vec<u8>, Error> {
        let filename = TERRA_DIRECTORY.join("assets").join(name);
        if filename.exists() {
            Ok(tokio::fs::read(&filename).await?)
        } else {
            let contents = Self::download(&self.server, &format!("assets/{}", name)).await?;
            if self.server.starts_with("http://") || self.server.starts_with("https://") {
                if let Some(parent) = filename.parent() {
                    fs::create_dir_all(parent)?;
                }
                AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                    .write(|f| f.write_all(&contents))?;
            }
            Ok(contents)
        }
    }

    async fn download(server: &str, path: &str) -> Result<Vec<u8>, Error> {
        match server.split_once("//") {
            Some(("file:", base_path)) => {
                let full_path = PathBuf::from(base_path).join(path);
                Ok(tokio::fs::read(&full_path).await?)
            }
            Some(("http:", ..)) | Some(("https:", ..)) => {
                let url = format!("{}{}", server, path);
                let client = hyper::Client::builder()
                    .build::<_, hyper::Body>(hyper_tls::HttpsConnector::new());
                let resp = client.get(url.parse()?).await?;
                if resp.status().is_success() {
                    Ok(hyper::body::to_bytes(resp.into_body()).await?.to_vec())
                } else {
                    Err(anyhow::format_err!(
                        "Tile download failed with {:?} for URL '{}'",
                        resp.status(),
                        url
                    ))
                }
            }
            _ => Err(anyhow::format_err!("Invalid server URL {}", server)),
        }
    }
}
