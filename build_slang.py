import argparse
import os
from pathlib import Path
import subprocess
import shutil


def delete_spv(path):
    print('Deleting spv from ' + str(path))

    for spv in path.glob('*.spv'):
        os.remove(spv)


def build_slang(shader_path, build_type):
    print('Building Slang...')

    skip_files = ['common.slang']

    for slang in shader_path.glob('*.slang'):
        for skip in skip_files:
            if not slang.match(skip):
                spv_name = str(slang) + '.spv'

                cmd = 'slangc ' + str(slang)
                if (build_type == 'Debug'):
                    cmd += ' -g3 -O0'
                elif (build_type == 'Release'):
                    cmd += ' -g0 -O3'

                cmd += ' -matrix-layout-column-major -o ' + str(spv_name)

                subprocess.call(cmd)


def move_spv(src, dst):
    print('Move spv to ' + str(dst))

    if not dst.exists():
        os.mkdir(dst)

    for src_glsl in src.glob('*.spv'):
        shutil.move(src_glsl, dst)


def main(args):
    local_shader_path = Path(args.source_dir) / 'shaders'
    delete_spv(local_shader_path)
    build_slang(local_shader_path, args.build_type)

    remote_shader_path = Path(args.output_dir) / 'shaders'
    delete_spv(remote_shader_path)
    move_spv(local_shader_path, remote_shader_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('source_dir')
    parser.add_argument('build_type')
    parser.add_argument('output_dir')

    main(parser.parse_args())
