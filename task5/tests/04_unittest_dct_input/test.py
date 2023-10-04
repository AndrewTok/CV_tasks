import os
import numpy as np
import image_compression as ic


def test_dct_1():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    block_1 = np.array([
        [-76, -73, -67, -62, -58, -67, -64, -55],
        [-65, -69, -73, -38, -19, -43, -59, -56],
        [-66, -69, -60, -15, 16, -24, -62, -55],
        [-65, -70, -57, -6, 26, -22, -58, -59],
        [-61, -67, -60, -24, -2, -40, -60, -58],
        [-49, -63, -68, -58, -51, -60, -70, -53],
        [-43, -57, -64, -69, -73, -67, -63, -45],
        [-41, -49, -59, -60, -63, -52, -50, -34]
    ])
    answer = ic.dct(block_1)
    true_answer = np.array([
        [-415.3749999999999, -30.185717276809033, -61.1970619502957, 27.23932249600452, 56.124999999999964, -20.095173772334842, -2.387647095293558, 0.46181544244846645],
        [4.4655237014136855, -21.857439332259844, -60.75803811653402, 10.253636818417837, 13.145110120476232, -7.0874180078452005, -8.535436712969494, 4.8768884966804045],
        [-46.834484742312476, 7.370597353426694, 77.12938757875553, -24.561982249733376, -28.911688429320662, 9.933520952775087, 5.416815472394543, -5.648950862137469],
        [-48.53496666553105, 12.068360940019197, 34.09976717271505, -14.759411080801929, -10.240606801750438, 6.295967438373016, 1.8311650530957317, 1.945936514864812],
        [12.12499999999995, -6.553449928892075, -13.196120970971862, -3.951427727907836, -1.8749999999999893, 1.7452844510267367, -2.7872282503369483, 3.1352823039767697],
        [-7.7347436775991625, 2.905461382890558, 2.379795764875581, -5.939313935865533, -2.37779670673259, 0.9413915961413784, 4.303713343622748, 1.8486910259091216],
        [-1.030674013497251, 0.18306744355204074, 0.41681547239454186, -2.4155613745353888, -0.8777939199423077, -3.0193065522845317, 4.120612421244484, -0.6619484539385858],
        [-0.16537560203663063, 0.14160712244184515, -1.0715363895103496, -4.192912078044711, -1.170314092006254, -0.09776107933753686, 0.5012693916445825, 1.6754588169203766]
    ]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5


def test_dct_2():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    block_2 = np.array([
        [11, 16, 21, 25, 27, 27, 27, 27],
        [16, 23, 25, 28, 31, 28, 28, 28],
        [22, 27, 32, 35, 30, 28, 28, 28],
        [31, 33, 34, 32, 32, 31, 31, 31],
        [31, 32, 33, 34, 34, 27, 27, 27],
        [33, 33, 33, 33, 32, 29, 29, 29],
        [34, 34, 33, 35, 34, 29, 29, 29],
        [34, 34, 34, 33, 35, 30, 30, 30]
    ])
    answer = ic.dct(block_2)
    true_answer = np.array([
        [235.74999999999994, -0.9351164036280837, -12.148549234796334, -5.376277613016528, 2.00000000000001, -1.6379429700024482, -2.544651545836887, 1.4708294144672716],
        [-22.763816029339623, -17.62040876165767, -6.146643044326955, -2.9168901917728363, -2.682285339244544, -0.1172912853942556, 0.20763328608220366, -1.3894572686682756],
        [-10.78594357877394, -9.134075714407516, -1.6642135623730958, 1.3035586141888276, 0.03962816694527849, -0.8968036370943202, -0.3535533905932722, 0.12912070468827275],
        [-7.228539557385993, -2.0226609747637454, 0.304337078846439, 1.6577622512496148, 1.0432357677505237, -0.12042715179609464, -0.23433527975292456, 0.15870288997572257],
        [-0.4999999999999839, -0.7398964891217092, 1.4022338229676485, 1.3829016372511624, -0.25000000000000133, -0.626498635445897, 0.7721659832740202, 1.4221939765228393],
        [1.655869951597331, -0.28002323353535363, 1.6736384942739813, -0.20621358942299528, -0.6773262739250905, 1.4488394665485922, 0.9126844218827257, -1.1084474123906483],
        [-1.2148749382159463, -0.30680091326853587, -0.35355339059327084, -1.5539311107039464, -0.5576106243469137, 1.7535069974989999, 1.164213562373096, -0.6818000993591055],
        [-2.63438614350321, 1.5247562937048196, -3.7441118317089392, -1.7969222769788422, 1.9061043020806887, 1.2044301517868075, -0.6129449823876908, -0.4861929561405433]
    ]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5