use crate::cache::{AssetLoadContext, GeneratedAsset};
use crate::sky::lut::{LookupTable, LookupTableDefinition};
use crate::sky::precompute::{InscatteringTable, TransmittanceTable};
use anyhow::Error;

mod lut;
mod precompute;

pub(crate) struct Atmosphere {
    pub transmittance: LookupTable,
    pub inscattering: LookupTable,
}
impl Atmosphere {
    pub fn new(context: &mut AssetLoadContext) -> Result<Self, Error> {
        let transmittance = TransmittanceTable { steps: 1000 }.generate(context)?;
        let inscattering =
            InscatteringTable { steps: 30, transmittance: &transmittance }.generate(context)?;

        Ok(Self { transmittance, inscattering })
    }
}
