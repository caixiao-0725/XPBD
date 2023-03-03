# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)
project ("XPBD")
find_package(CUDA REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/Include)
link_directories(lib)
CUDA_add_executable (XPBD  Cloth.cu glad.c Model.cu XPBD.cu)
target_include_directories(XPBD PUBLIC ${PROJECT_BINARY_DIR})
target_link_libraries(XPBD glfw3  opengl32 kernel32.lib
user32.lib
gdi32.lib
winspool.lib
comdlg32.lib
advapi32.lib
shell32.lib
ole32.lib
oleaut32.lib
uuid.lib
odbc32.lib
odbccp32.lib)

