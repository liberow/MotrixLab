"""
URDF 转 MJCF 工具

将 ROS 格式的 URDF 文件转换为 MuJoCo 的 MJCF 格式。

注意：MuJoCo 的 URDF 解析器在处理 mesh 路径时只使用文件名，
忽略目录结构。因此需要在模型目录创建 mesh 文件的符号链接。
"""
import mujoco
import os
import glob

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "sektion_cabinet_model")

urdf_path = os.path.join(model_dir, "urdf/sektion_cabinet_2.urdf")
output_path = os.path.join(model_dir, "mjcf/sektion_cabinet_2.xml")

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 在模型目录创建 mesh 文件的符号链接（MuJoCo URDF 解析器需要）
meshes_dir = os.path.join(model_dir, "meshes")
for mesh_file in glob.glob(os.path.join(meshes_dir, "*.obj")) + glob.glob(os.path.join(meshes_dir, "*.stl")):
    link_path = os.path.join(model_dir, os.path.basename(mesh_file))
    if not os.path.exists(link_path):
        os.symlink(mesh_file, link_path)

# 读取 URDF 文件并处理 ROS package:// 路径
with open(urdf_path, 'r') as f:
    urdf_content = f.read()

# 移除 package:// 前缀和 meshes/ 目录（因为符号链接在模型根目录）
urdf_content = urdf_content.replace('package://sektion_cabinet_model/', '')
urdf_content = urdf_content.replace('meshes/', '')

# 创建临时 URDF 文件
temp_urdf_path = os.path.join(model_dir, "_temp_converted.urdf")
with open(temp_urdf_path, 'w') as f:
    f.write(urdf_content)

# 保存原始工作目录，切换到模型目录
original_dir = os.getcwd()
os.chdir(model_dir)

try:
    # 加载 URDF 并导出为 MJCF
    model = mujoco.MjModel.from_xml_path("_temp_converted.urdf")
    mujoco.mj_saveLastXML(output_path, model)
    print(f"成功转换: {urdf_path} -> {output_path}")
finally:
    os.chdir(original_dir)
    # 清理临时文件
    if os.path.exists(temp_urdf_path):
        os.remove(temp_urdf_path)
