// Collection of hash related functions, courtesy of [Spatial](https://stackoverflow.com/a/17479300)

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

float random(uint v)  { return floatConstruct(hash(v)); }
float random(uvec2 v) { return floatConstruct(hash(v)); }
float random(uvec3 v) { return floatConstruct(hash(v)); }
float random(uvec4 v) { return floatConstruct(hash(v)); }

vec2 box_muller_transform(float u1, float u2) {
	float r = sqrt(-2.0 * log(1-u1));
	float theta = 2.0*3.14159265*u2;
    return vec2(r*sin(theta), r*cos(theta));
}

vec2 guassian_random(float v) {
	float r = random(v);
	return box_muller_transform(r, random(r));
}
vec2 guassian_random(vec2 v) {
	float r = random(v);
	return box_muller_transform(r, random(r));
}
vec2 guassian_random(vec3 v) {
	float r = random(v);
	return box_muller_transform(r, random(r));
}

vec3 dither(vec2 fragcoord) {
	 return vec3(random(fragcoord) - 0.5,
				 random(fragcoord + vec2(1.2345, 6.7890)) - 0.5,
				 random(fragcoord + vec2(6.7890, 1.2345)) - 0.5)
		* 0.00392156862;
}
