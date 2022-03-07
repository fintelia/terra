use std::f64::consts::PI;

use geo::prelude::*;
use mint::{ColumnMatrix3, ColumnMatrix4, Vector3};

#[derive(Clone, Debug)]
struct PlanetCam {
    latitude: f64,
    longitude: f64,
    bearing: f64,
    pitch: f64,
    height: f64,
}

impl PlanetCam {
    fn move_forward(&mut self, meters: f64) {
        if meters == 0.0 {
            return;
        }

        let start = geo::Point::new(self.longitude, self.latitude);
        let end = start.haversine_destination(self.bearing, meters);
        let new_bearing =
            if meters > 0.0 { end.bearing(start) + 180.0 } else { end.bearing(start) };

        assert_eq!(start.y(), self.latitude);
        assert_eq!(start.x(), self.longitude);

        self.latitude = end.y();
        self.longitude = end.x();
        self.bearing = new_bearing;
    }
    fn move_right(&mut self, meters: f64) {
        if meters == 0.0 {
            return;
        }

        let start = geo::Point::new(self.longitude, self.latitude);
        let end = start.haversine_destination(self.bearing + 90.0, meters);
        let new_bearing =
            if meters > 0.0 { end.bearing(start) + 90.0 } else { end.bearing(start) - 90.0 };

        self.latitude = end.y().min(89.999).max(-89.999);
        self.longitude = end.x();
        self.bearing = new_bearing;
    }
    fn move_up(&mut self, meters: f64) {
        self.height = (self.height + meters).max(0.001);
    }

    fn increase_bearing(&mut self, degrees: f64) {
        self.bearing += degrees;
        if self.bearing >= 360.0 {
            self.bearing -= 360.0;
        }
        if self.bearing < 0.0 {
            self.bearing += 360.0;
        }
    }
    fn increase_pitch(&mut self, degrees: f64) {
        self.pitch += degrees;
        self.pitch = self.pitch.min(89.0).max(-89.0);
    }

    /// Returns the ECEF position and the view matrix associated with this camera.
    fn position_view(&self, terrain_elevation: f64) -> (Vector3<f64>, ColumnMatrix3<f32>) {
        let r = 6371000.0 + self.height + terrain_elevation;
        let lat = self.latitude.to_radians();
        let long = self.longitude.to_radians();

        let up = cgmath::Vector3::new(lat.cos() * long.cos(), lat.cos() * long.sin(), lat.sin());
        let position = up * r;

        let adjusted_pitch =
            (self.pitch.to_radians() - f64::acos(6371000.0 / r)).clamp(-0.499 * PI, 0.499 * PI);

        let start = geo::Point::new(self.longitude, self.latitude);
        let center = start.haversine_destination(self.bearing, 1.0);
        let latc = center.y().to_radians();
        let longc = center.x().to_radians();
        let forward = (1.0 + adjusted_pitch.tan() / 6371000.0)
            * cgmath::Vector3::new(latc.cos() * longc.cos(), latc.cos() * longc.sin(), latc.sin())
            - up;

        let matrix = cgmath::Matrix3::look_to_rh(forward, up);
        (position.into(), matrix.cast().unwrap().into())
    }
}

pub struct DualPlanetCam {
    anchored: Option<PlanetCam>,
    free: PlanetCam,
}
impl DualPlanetCam {
    pub fn new(latitude: f64, longitude: f64, bearing: f64, pitch: f64, height: f64) -> Self {
        Self { anchored: None, free: PlanetCam { latitude, longitude, bearing, pitch, height } }
    }

    pub fn detach(&mut self) {
        self.anchored = Some(self.free.clone());
    }
    pub fn attach(&mut self) {
        self.anchored = None;
    }
    pub fn is_detached(&self) -> bool {
        self.anchored.is_some()
    }

    pub fn move_forward(&mut self, meters: f64) {
        self.free.move_forward(meters);
    }
    pub fn move_right(&mut self, meters: f64) {
        self.free.move_right(meters);
    }
    pub fn move_up(&mut self, meters: f64) {
        self.free.move_up(meters);
    }

    pub fn increase_bearing(&mut self, degrees: f64) {
        self.free.increase_bearing(degrees);
    }
    pub fn increase_pitch(&mut self, degrees: f64) {
        self.free.increase_pitch(degrees);
    }

    pub fn latitude_longitude(&self) -> (f64, f64) {
        (self.free.latitude, self.free.longitude)
    }
    pub fn height(&self) -> f64 {
        self.free.height
    }

    pub fn anchored_latitude_longitude(&self) -> (f64, f64) {
        let c = self.anchored.as_ref().unwrap_or(&self.free);
        (c.latitude, c.longitude)
    }

    pub fn anchored_position_view(
        &self,
        terrain_elevation: f64,
    ) -> (Vector3<f64>, ColumnMatrix3<f32>) {
        match &self.anchored {
            Some(a) => a.position_view(terrain_elevation),
            None => self.free.position_view(terrain_elevation),
        }
    }
    pub fn free_position_view(&self, terrain_elevation: f64) -> ColumnMatrix4<f32> {
        match &self.anchored {
            Some(a) => {
                let anchored = a.position_view(terrain_elevation);
                let free = self.free.position_view(terrain_elevation);

                let look_at = cgmath::Matrix4::from(cgmath::Matrix3::from(free.1));
                let translation = cgmath::Vector3::from(anchored.0) - cgmath::Vector3::from(free.0);

                (look_at * cgmath::Matrix4::from_translation(translation.cast().unwrap())).into()
            }
            None => cgmath::Matrix4::from(cgmath::Matrix3::from(
                self.free.position_view(terrain_elevation).1,
            ))
            .into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use cgmath::{assert_abs_diff_eq, assert_relative_eq, MetricSpace};
    use geo::prelude::HaversineDestination;

    use crate::PlanetCam;

    #[test]
    fn it_works() {
        let camera =
            PlanetCam { latitude: 23.0, longitude: 45.0, height: 100.0, bearing: 67.0, pitch: 0.0 };
        let mut camera2 = camera.clone();
        for x in 1..100 {
            let mut full = camera.clone();
            full.move_forward(10000.0 * x as f64);

            camera2.move_forward(10000.0);

            assert_abs_diff_eq!(full.latitude, camera2.latitude, epsilon = 0.0001);
            assert_abs_diff_eq!(full.longitude, camera2.longitude, epsilon = 0.0001);
        }
    }

    #[test]
    fn move_distance() {
        let camera =
            PlanetCam { latitude: 23.0, longitude: 45.0, height: 100.0, bearing: 67.0, pitch: 0.0 };
        let mut camera2 = camera.clone();
        camera2.move_forward(100.0);

        const MEAN_EARTH_RADIUS: f64 = 6371008.8;
        const MEAN_EARTH_CIRCUMFERENCE: f64 = MEAN_EARTH_RADIUS * std::f64::consts::PI;

        let (lat, long) = (camera.latitude.to_radians(), camera.longitude.to_radians());
        let position = MEAN_EARTH_RADIUS
            * cgmath::Vector3::new(lat.cos() * long.cos(), lat.cos() * long.sin(), lat.sin());

        let (lat, long) = (camera2.latitude.to_radians(), camera2.longitude.to_radians());
        let position2 = MEAN_EARTH_RADIUS
            * cgmath::Vector3::new(lat.cos() * long.cos(), lat.cos() * long.sin(), lat.sin());

        let distance = position.distance(position2);
        assert_abs_diff_eq!(distance, 100.0, epsilon = 0.1);

        let start = geo::Point::new(camera.longitude, 0.0);
        let end = start.haversine_destination(0.0, 1000.0);
        assert_abs_diff_eq!(
            end.y() - start.y(),
            1000.0 * 180.0 / MEAN_EARTH_CIRCUMFERENCE,
            epsilon = 0.0000001
        );

        let end = start.haversine_destination(90.0, 1000.0);
        assert_abs_diff_eq!(
            end.x() - start.x(),
            1000.0 * 180.0 / MEAN_EARTH_CIRCUMFERENCE,
            epsilon = 0.0000001
        );
    }
}
