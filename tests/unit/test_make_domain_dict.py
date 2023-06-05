from src.utils.ndo_score import make_domain_dict

def test_make_domain_dict1():
    domain_string = '1-3_11-12,4-6,9,8-8'
    n_res = 14
    domain_dict = make_domain_dict(domain_string, n_res)
    assert sorted(domain_dict['linker']) == [0, 7, 10, 13]
    expected_domains ={
        '1-2-3-11-12',
        '4-5-6',
        '9',
        '8'
    }
    for k,v in domain_dict.items():
        if k == 'linker':
            continue
        domain_hash = '-'.join([str(i) for i in sorted(v)])
        assert domain_hash in expected_domains
        expected_domains.remove(domain_hash)