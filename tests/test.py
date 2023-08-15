
def main(): 
    try: 
        from stoned_selfies.source import get_fingerprint, get_fp_scores
        print('All Good!')

    except: 
        raise ImportError()


if __name__ == '__main__':
    main()
