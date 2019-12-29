// From: https://github.com/SaschaWillems/Vulkan-glTF-PBR/blob/master/data/shaders/pbr_khr.frag

struct PBRInfo
{
	float NdotL;                  // cos angle between normal and light direction
	float NdotV;                  // cos angle between normal and view direction
	float NdotH;                  // cos angle between normal and half vector
	float LdotH;                  // cos angle between light direction and half vector
	float VdotH;                  // cos angle between view direction and half vector
	float perceptualRoughness;    // roughness value, as authored by the model creator (input to shader)
	float alphaRoughness;         // roughness mapped to a more linear change in the roughness (proposed by [2])
	vec3 diffuseColor;            // color contribution from diffuse lighting
};

const float M_PI = 3.141592653589793;
const float c_MinRoughness = 0.04;

const float PBR_WORKFLOW_METALLIC_ROUGHNESS = 0.0;
const float PBR_WORKFLOW_SPECULAR_GLOSINESS = 1.0f;

#define MANUAL_SRGB 1

vec3 Uncharted2Tonemap(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	float W = 11.2;
	return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

vec4 tonemap(vec4 color, float exposure, float gamma)
{
	vec3 outcol = Uncharted2Tonemap(color.rgb * exposure);
	outcol = outcol * (1.0f / Uncharted2Tonemap(vec3(11.2f)));
	return vec4(pow(outcol, vec3(1.0f / gamma)), color.a);
}

// vec4 SRGBtoLINEAR(vec4 srgbIn)
// {
// 	#ifdef MANUAL_SRGB
// 	#ifdef SRGB_FAST_APPROXIMATION
// 	vec3 linOut = pow(srgbIn.xyz,vec3(2.2));
// 	#else //SRGB_FAST_APPROXIMATION
// 	vec3 bLess = step(vec3(0.04045),srgbIn.xyz);
// 	vec3 linOut = mix( srgbIn.xyz/vec3(12.92), pow((srgbIn.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
// 	#endif //SRGB_FAST_APPROXIMATION
// 	return vec4(linOut,srgbIn.w);;
// 	#else //MANUAL_SRGB
// 	return srgbIn;
// 	#endif //MANUAL_SRGB
// }

// // Calculation of the lighting contribution from an optional Image Based Light source.
// // Precomputed Environment Maps are required uniform inputs and are computed as outlined in [1].
// // See our README.md on Environment Maps [3] for additional discussion.
// vec3 getIBLContribution(PBRInfo pbrInputs, vec3 n, vec3 reflection)
// {
// 	float lod = (pbrInputs.perceptualRoughness * uboParams.prefilteredCubeMipLevels);
// 	// retrieve a scale and bias to F0. See [1], Figure 3
// 	vec3 brdf = (texture(samplerBRDFLUT, vec2(pbrInputs.NdotV, 1.0 - pbrInputs.perceptualRoughness))).rgb;
// 	vec3 diffuseLight = SRGBtoLINEAR(tonemap(texture(samplerIrradiance, n))).rgb;
// 	vec3 specularLight = SRGBtoLINEAR(tonemap(textureLod(prefilteredMap, reflection, lod))).rgb;
// 	vec3 diffuse = diffuseLight * pbrInputs.diffuseColor;
// 	vec3 specular = specularLight * (vec3(0.04) * brdf.x + brdf.y);
// 	return diffuse + specular;
// }

// Basic Lambertian diffuse. Implementation from Lambert's Photometria
// https://archive.org/details/lambertsphotome00lambgoog See also [1], Equation
// 1
vec3 diffuse(PBRInfo pbrInputs)
{
	return pbrInputs.diffuseColor / M_PI;
}

// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
float specularReflection(PBRInfo pbrInputs)
{
	return 0.04 + 0.96 * pow(clamp(1.0 - pbrInputs.VdotH, 0.0, 1.0), 5.0);
}

// This calculates the specular geometric attenuation (aka G()), where rougher
// material will reflect less light back to the viewer.  This implementation is
// based on [1] Equation 4, and we adopt their modifications to alphaRoughness
// as input as originally proposed in [2].
float geometricOcclusion(PBRInfo pbrInputs)
{
	float NdotL = pbrInputs.NdotL;
	float NdotV = pbrInputs.NdotV;
	float r = pbrInputs.alphaRoughness;

	float attenuationL = 2.0 * NdotL / (NdotL + sqrt(r * r + (1.0 - r * r) * (NdotL * NdotL)));
	float attenuationV = 2.0 * NdotV / (NdotV + sqrt(r * r + (1.0 - r * r) * (NdotV * NdotV)));
	return attenuationL * attenuationV;
}

// The following equation(s) model the distribution of microfacet normals across
// the area being drawn (aka D()) Implementation from "Average Irregularity
// Representation of a Roughened Surface for Ray Reflection" by
// T. S. Trowbridge, and K. P. Reitz Follows the distribution function
// recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation
// 3.
float microfacetDistribution(PBRInfo pbrInputs)
{
	float roughnessSq = pbrInputs.alphaRoughness * pbrInputs.alphaRoughness;
	float f = (pbrInputs.NdotH * roughnessSq - pbrInputs.NdotH) * pbrInputs.NdotH + 1.0;
	return roughnessSq / (M_PI * f * f);
}

vec3 pbr(vec3 albedo,
		 float perceptualRoughness,
		 vec3 position,
		 vec3 normal,
		 vec3 camera,
		 vec3 lightDir,
		 vec3 lightColor) {

	vec3 n = normalize(normal);
	vec3 v = normalize(camera - position);
	vec3 l = normalize(lightDir);
	vec3 h = normalize(l+v);
	float NdotL = clamp(dot(n, l), 0.001, 1.0);
	float NdotV = clamp(abs(dot(n, v)), 0.001, 1.0);
	float NdotH = clamp(dot(n, h), 0.0, 1.0);
	float LdotH = clamp(dot(l, h), 0.0, 1.0);
	float VdotH = clamp(dot(v, h), 0.0, 1.0);

	float alphaRoughness = perceptualRoughness * perceptualRoughness;

	PBRInfo pbrInputs = PBRInfo(
		NdotL,
		NdotV,
		NdotH,
		LdotH,
		VdotH,
		perceptualRoughness,
		alphaRoughness,
		albedo
	);

	float F = specularReflection(pbrInputs);
	float G = geometricOcclusion(pbrInputs);
	float D = microfacetDistribution(pbrInputs);

	// Calculation of analytical lighting contribution
	vec3 diffuseContrib = (1.0 - F) * diffuse(pbrInputs);
	float specContrib = F * G * D / (4.0 * NdotL * NdotV);
	// Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
	vec3 color = NdotL * lightColor * (diffuseContrib + vec3(specContrib));

	// vec3 reflection = -normalize(reflect(v, n));
	// reflection.y *= -1.0f;
	// color += getIBLContribution(pbrInputs, n, reflection);

	return color;
}
