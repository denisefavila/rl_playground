from envs.sys_admin import SysAdminEnv
from iteration.value_iteration import value_iteration


def main():

    sys_admin_env = SysAdminEnv()

    policy, value_function = value_iteration(env=sys_admin_env, discount_factor=0.9)

    print(f'Value Function: {value_function}')
    print(f'Policy: {policy}')


if __name__ == "__main__":

    main()
