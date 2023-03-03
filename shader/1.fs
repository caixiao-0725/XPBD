#version 330 core
in vec3 normal;
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec4 FragPosLightSpace;
} fs_in;


uniform vec3 lightPos;
uniform vec3 viewPos;

uniform sampler2D shadowMap;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    float bias = 0.005;
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // check whether current frag pos is in shadow
    float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;

    if(projCoords.z > 1.0)
        shadow = 0.0;
    return shadow;
}

void main()
{
    vec3 color = normal;
    float shadow = ShadowCalculation(fs_in.FragPosLightSpace);     
    vec3 lighting = (1.0 - shadow)*color;                
    FragColor = vec4(lighting,1.0);
    
}