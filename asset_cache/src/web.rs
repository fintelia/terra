use failure::Error;

use std::fs::{self, File};
use std::path::PathBuf;
use std::io::Write;
use std::sync::Arc;

use ::*;

pub trait WebAssetDefinition: 'static {
    fn url(&self) -> String;
    fn filename(&self) -> String;
    fn parse(&self, context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Arc<Asset>, Error>;
    fn credentials(&self) -> Option<(String, String)> {
        None
    }
    fn load(&self, context: &mut AssetLoadContext) -> Result<Arc<Asset>, Error> {
        context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
        let ret = (|| {
            let filename: PathBuf = context.directory().join(self.filename());

            if let Ok(file) = File::open(&filename) {
                if let Ok(data) = asset::read_file(context, file) {
                    context.reset(&format!("Parsing {}... ", &self.filename()), 100);
                    if let Ok(asset) = self.parse(context, data) {
                        return Ok(asset);
                    }
                }
            }

            let mut data = Vec::<u8>::new();
            {
                context.reset(&format!("Downloading {}... ", &self.filename()), 100);
                // Bytes display will be disabled by the reset() below, or in the event of an error,
                // by the decrement_level() call in the outer scope.
                context.bytes_display_enabled(true);

                use curl::easy::Easy;
                let mut easy = Easy::new();
                easy.url(&self.url())?;
                easy.follow_location(true)?;
                if let Some((username, password)) = self.credentials() {
                    easy.cookie_file("")?;
                    easy.unrestricted_auth(true)?;
                    easy.username(&username)?;
                    easy.password(&password)?;
                }
                let mut easy = easy.transfer();
                easy.write_function(|d| {
                    let len = d.len();
                    data.extend(d);
                    Ok(len)
                })?;
                easy.progress_function(|c, t, _, _| {
                    if t > 0.0 {
                        context.set_progress_and_total(c, t);
                    }
                    true
                })?;
                easy.perform()?;
            }

            context.reset(&format!("Saving {}... ", &self.filename()), 100);
            if let Some(parent) = filename.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(&filename)?;
            file.write_all(&data)?;
            file.sync_all()?;
            context.reset(&format!("Parsing {}... ", &self.filename()), 100);
            Ok(self.parse(context, data)?)
        })();
        context.decrement_level();
        ret
    }
}
