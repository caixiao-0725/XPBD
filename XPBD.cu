#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>    
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h> 
#include <glm/glm/glm.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>
#include <glm/glm/gtc/type_ptr.hpp>

#include "Geometry.cuh"
#include "Shader.cuh"
#include "Camera.cuh"
#include "Cloth.cuh"
#include "Plane.h"
#include "matrix.h"
#include "Model.cuh"
#include<Windows.h>
#include<algorithm>
typedef unsigned int uint;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

#define stiffness0  1.0f
#define stiffness1  50.f
#define mass  1.0f
#define dt 0.0167f
#define n 21
#define STEPS 10 

float sdt = dt / STEPS;
float damp = 0.999f;
// camera
Camera camera(glm::vec3(0.0f, 0.5f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;


cudaGraphicsResource* resource;

GLFWwindow* window;

void GLFWInit() {

    // glfw: initialize and configure
 // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    // glfw window creation
    // --------------------
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSwapInterval(1);
    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }
}



int main()
{
    GLFWInit();
    uint shaderId = createShader("E:/try_for_chunzhao/XPBD/shader/1.vs", "E:/try_for_chunzhao/XPBD/shader/1.fs");
    uint shadowShader = createShader("E:/try_for_chunzhao/XPBD/shader/shadow.vs", "E:/try_for_chunzhao/XPBD/shader/shadow.fs");
    Plane ground = Plane(vec3f(10, -0.2f, 10), vec3f(-10, -0.2f, 10), vec3f(10, -0.2f, -10), vec3f(-10, -0.2f, -10), vec3f(1, 1, 1));
    Model teapot = Model("E:/try_for_chunzhao/XPBD/model/teapot.obj");

    std::vector<vec3f> normals_(teapot.nverts());
    
    /*
    size_t size;
    vec3f* X_device;
    vec3f* V_device;
    cudaMalloc((void**)&V_device, sizeof(vec3f) * n*n);
    */
    
    for (int i = 0; i < teapot.nfaces(); i++) {
        vec3i temp = teapot.faces_[i];
        vec3f AO = teapot.verts_[temp.y] - teapot.verts_[temp.x];
        vec3f BO = teapot.verts_[temp.z] - teapot.verts_[temp.x];
        vec3f N = -1.0f * normalize(AO ^ BO);
        normals_[temp.x] += N;
        normals_[temp.y] += N;
        normals_[temp.z] += N;
    }
    
    Cloth MeshObj = Cloth(teapot.verts_.data(), normals_.data(), teapot.nverts(), teapot.faces_.data(), teapot.nfaces());

    //写个衣服布料  位置，uv，边初始长度
    int edge_num = n * (n - 1) * 2;
    std::vector<vec3f> X(n * n);
    std::vector<vec3f> P(n * n);
    std::vector<vec3f> V(n * n);
    std::vector<vec3f> X_color(n * n);
    std::vector<vec3i> T(2 * (n - 1) * (n - 1));
    std::vector<int> Edge(2 * edge_num);
    std::vector<float> m_Lambda(edge_num);
    int count_edge = 0;
    for (int i = 0; i < n ; i++) {
        for (int j = 0; j < n-1; j++) {
            Edge[count_edge] = n * i + j;
            Edge[count_edge + 1] = n * i + j + 1;
            count_edge += 2;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++) {
            Edge[count_edge] = i + j*n;
            Edge[count_edge+1] = i + (j+1)*n;
            count_edge += 2;
        }
    }



    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            X[j * n + i] = vec3f(2.5 - 5.0f * i / (n - 1), 5.0f, 2.5f - 5.0f * j / (n - 1));
            X_color[j * n + i] = vec3f(1.0f, 0.0f, 0.0f);
        }
    }

    for (int j = 0; j < n - 1; j++) {
        for (int i = 0; i < n - 1; i++)
        {
            T[j * (n - 1) + i] = vec3i(j * n + i, j * n + i + 1, (j + 1) * n + i + 1);
            T[(n - 1) * (n - 1) + j * (n - 1) + i] = vec3i(j * n + i, (j + 1) * n + i + 1, (j + 1) * n + i);
        }
    }
    for (int i = 0; i < X.size(); i++) {
        P[i] = X[i];
    }
    float restLen = 5.0f / (n - 1);
    float w0 = 1, w1 = 1;
    float w = w0 + w1;
    float alpha = 0.001f / dt / dt;

    Cloth clo = Cloth(X.data(), X_color.data(), X.size(), T.data(), T.size());
    // lighting info
    // -------------
    glm::vec3 lightPos(-2.0f, 4.0f, -1.0f);
    unsigned int depthMapFBO;
    unsigned int depthMap;
    //设置阴影
    glGenFramebuffers(1, &depthMapFBO);
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    GLfloat borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    // attach depth texture as FBO's depth buffer
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glUseProgram(shaderId);
    shaderSetInt(shaderId, "shadowMap", 0);

    glEnable(GL_DEPTH_TEST);
    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {

        /*
        cudaGraphicsGLRegisterBuffer(&resource, clo.vertices_VBO, cudaGraphicsMapFlagsNone);

        cudaGraphicsMapResources(1, &resource, NULL);
        cudaGraphicsResourceGetMappedPointer((void**)&X_device, &size, resource);

        Particle <<< (X.size() - 1) / 1024 + 1, 1024 >> > (X_device, V_device, X.size());
        //PBD <<<(cube.neles() - 1) / 1024 + 1, 1024 >>> (Force_device,X_device,eles_device,Dm_inv_device,volume_device, cube.neles());
        //Update_V <<< (cube.nverts() - 1) / 1024 + 1, 1024 >>> (Force_device, X_device, V_device, cube.nverts());

        cudaGraphicsUnmapResources(1, &resource, NULL);
        */
        for (int i = 0; i < edge_num; i++) {
            m_Lambda[i] = 0;
        }
        //presolve
        for (int i = 0; i < X.size(); i++) {
            if (i == 0 || i == 20) continue;
            V[i] += vec3f(0, -9.8f, 0) * dt;
            P[i] = X[i] + V[i] * dt;
        }
        for (int iter = 0; iter < STEPS; iter++) {
            //distance constraint
            //XPBD
          
            for (int i = 0; i < edge_num; i++) {
                int id0 = Edge[2 * i];
                int id1 = Edge[2 * i + 1];
                vec3f grad = P[id0] - P[id1];
                float length = grad.Length();
                float C = length - restLen;
                float normalizingFactor = 1.0 / length;
                float dlambda = (-C - alpha*m_Lambda[i]) / (w + alpha);
                vec3f correction_vector = dlambda * grad * normalizingFactor;
                              m_Lambda[i] += dlambda;
                
                if (id0 != 0 && id0 != 20) {
                    P[id0] += correction_vector * w0;
                }
                if (id1 != 0 && id1 != 20) {
                    P[id1] -= correction_vector * w1;
                }
            }
            
            //PBD
            /*
            for (int i = 0; i < edge_num; i++) {
                int id0 = Edge[2 * i];
                int id1 = Edge[2 * i + 1];
                vec3f grad = P[id0] - P[id1];
                float length = grad.Length();
                float C = length - restLen;
                
                vec3f correction_vector = stiffness0*grad/length*(-C)/w;
                if (id0 != 0 && id0 != 20) {
                    P[id0] += correction_vector * w0;
                }
                if (id1 != 0 && id1 != 20) {
                    P[id1] -= correction_vector * w1;
                }
            }*/
        }
        for (int i = 0; i < X.size(); i++) {
            if (i == 0 || i == 20) continue;
            V[i] = (P[i]-X[i]) / dt;
            X[i] = P[i];
        }



        clo = Cloth(X.data(), X_color.data(), X.size(), T.data(), T.size());
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        // input
        // -----
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glm::mat4 lightProjection, lightView;
        glm::mat4 lightSpaceMatrix;
        float near_plane = 1.0f, far_plane = 7.5f;
        lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
        lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));
        lightSpaceMatrix = lightProjection * lightView;

        glUseProgram(shadowShader);

        shaderSetMat4(shadowShader, "lightSpaceMatrix", lightSpaceMatrix);
        glViewport(0, 0, 1024, 1024);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);

        clo.genShadow(shadowShader);
        MeshObj.genShadow(shadowShader);
        ground.genShadow(shadowShader);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // reset viewport
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderId);

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        shaderSetMat4(shaderId, "projection", projection);
        // camera/view transformation
        glm::mat4 view = camera.GetViewMatrix();
        shaderSetMat4(shaderId, "view", view);

        shaderSetVec3(shaderId, "viewPos", camera.Position);
        shaderSetVec3(shaderId, "lightPos", lightPos);
        shaderSetMat4(shaderId, "lightSpaceMatrix", lightSpaceMatrix);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthMap);

        clo.draw(shaderId);
        //MeshObj.draw(shaderId);
        //ground.draw(shaderId);

        //Sleep(160);
        glBindVertexArray(0);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}