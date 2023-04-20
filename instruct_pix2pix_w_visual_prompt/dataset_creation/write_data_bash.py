import os


def main():
    cmd_tmpl = 'nohup python enrich_instruction_pairs.py --begin_num {} > nohup_{}.out &'
    cmds = []
    for i in range(0, 500000, 5000):
        cmd = cmd_tmpl.format(i,i)
        cmds.append(cmd)
    cmd_str = '\n'.join(cmds)
    print(cmd_str)


if __name__ == '__main__':
    main()