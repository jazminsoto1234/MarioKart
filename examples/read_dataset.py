from stk_agent.dataset import Episode


def main(path):
    ep = Episode(path)
    first = ep.transition(0, privileged=True)
    print('Episode:', ep.meta['episode_id'])
    print('RGB:', first['observation'].shape, first['observation'].dtype)
    print('Action sequence:', first['action_sequence'].shape)
    print('Phase:', first['phase'])
    print('Privileged keys:', sorted(first['privileged']))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        raise SystemExit('Uso: python examples/read_dataset.py <episode_dir>')
    main(sys.argv[1])
