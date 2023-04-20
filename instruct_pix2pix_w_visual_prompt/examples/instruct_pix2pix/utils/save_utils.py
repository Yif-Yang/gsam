import os
import sys


def save2data_drive(local_src_path, rel_dst_dir=None):
    rel_dst_dir = local_src_path if rel_dst_dir is None else rel_dst_dir

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    root_name = os.path.basename(root_path)
    dst_dir = os.path.join(root_name, rel_dst_dir)
    try:
        dst_path_templ = '\"https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/{}?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D\"'
        if os.path.isdir(local_src_path):
            dst_path = dst_dir
        else:
            fn = os.path.basename(local_src_path)
            dst_path = os.path.join(dst_dir, fn)

        network_dst_path = dst_path_templ.format(dst_path)
        cmd = 'touch {}/dummy.txt'.format(local_src_path)
        print(cmd)
        os.system(cmd)
        cmd = '~/azcopy copy --recursive {}/dummy.txt {}'.format(local_src_path,
                            dst_path_templ.format(os.path.join(local_src_path, 'dummy.txt')))
        print(cmd)
        os.system(cmd)
        cmd = '~/azcopy copy --recursive {} {} &'.format(local_src_path, network_dst_path)
        print(cmd)
        os.system(cmd)
    except Exception as ex:
        import traceback
        traceback.print_exc()
