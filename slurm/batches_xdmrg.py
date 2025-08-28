def get_xdmrg_batch_setup(projectname):
    batch_setup = {
        'xdmrg1-fse': {
            'description': "First batch in 2024 for xDMRG using forward subsystem expansion and deparallelization."
                           "The seeds encode the Hamiltonian parameters Eg: 16|00|00000"
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The next three second digit represents the g-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg1-fse',
            'batch': {
                'd-6.00|g0.00': {'seed_extent': [1000], 'seed_offset': [16000000000], },
                'd-5.00|g0.00': {'seed_extent': [1000], 'seed_offset': [15000000000], },
                'd-4.00|g0.00': {'seed_extent': [1000], 'seed_offset': [14000000000], },
                'd-3.00|g0.00': {'seed_extent': [1000], 'seed_offset': [13000000000], },
                'd-2.00|g0.00': {'seed_extent': [1000], 'seed_offset': [12000000000], },
                'd-1.00|g0.00': {'seed_extent': [1000], 'seed_offset': [11000000000], },
                'd+1.00|g0.00': {'seed_extent': [1000], 'seed_offset': [21000000000], },
                'd+2.00|g0.00': {'seed_extent': [1000], 'seed_offset': [22000000000], },
                'd+3.00|g0.00': {'seed_extent': [1000], 'seed_offset': [23000000000], },
                'd+4.00|g0.00': {'seed_extent': [1000], 'seed_offset': [24000000000], },
                'd+5.00|g0.00': {'seed_extent': [1000], 'seed_offset': [25000000000], },
                'd+6.00|g0.00': {'seed_extent': [1000], 'seed_offset': [26000000000], },

                'd-6.00|g0.01': {'seed_extent': [1000], 'seed_offset': [16001000000], },
                'd-5.00|g0.01': {'seed_extent': [1000], 'seed_offset': [15001000000], },
                'd-4.00|g0.01': {'seed_extent': [1000], 'seed_offset': [14001000000], },
                'd-3.00|g0.01': {'seed_extent': [1000], 'seed_offset': [13001000000], },
                'd-2.00|g0.01': {'seed_extent': [1000], 'seed_offset': [12001000000], },
                'd-1.00|g0.01': {'seed_extent': [1000], 'seed_offset': [11001000000], },
                'd+0.00|g0.01': {'seed_extent': [1000], 'seed_offset': [20001000000], },
                'd+1.00|g0.01': {'seed_extent': [1000], 'seed_offset': [21001000000], },
                'd+2.00|g0.01': {'seed_extent': [1000], 'seed_offset': [22001000000], },
                'd+3.00|g0.01': {'seed_extent': [1000], 'seed_offset': [23001000000], },
                'd+4.00|g0.01': {'seed_extent': [1000], 'seed_offset': [24001000000], },
                'd+5.00|g0.01': {'seed_extent': [1000], 'seed_offset': [25001000000], },
                'd+6.00|g0.01': {'seed_extent': [1000], 'seed_offset': [26001000000], },

                'd-6.00|g0.02': {'seed_extent': [1000], 'seed_offset': [16002000000], },
                'd-5.00|g0.02': {'seed_extent': [1000], 'seed_offset': [15002000000], },
                'd-4.00|g0.02': {'seed_extent': [1000], 'seed_offset': [14002000000], },
                'd-3.00|g0.02': {'seed_extent': [1000], 'seed_offset': [13002000000], },
                'd-2.00|g0.02': {'seed_extent': [1000], 'seed_offset': [12002000000], },
                'd-1.00|g0.02': {'seed_extent': [1000], 'seed_offset': [11002000000], },
                'd+0.00|g0.02': {'seed_extent': [1000], 'seed_offset': [20002000000], },
                'd+1.00|g0.02': {'seed_extent': [1000], 'seed_offset': [21002000000], },
                'd+2.00|g0.02': {'seed_extent': [1000], 'seed_offset': [22002000000], },
                'd+3.00|g0.02': {'seed_extent': [1000], 'seed_offset': [23002000000], },
                'd+4.00|g0.02': {'seed_extent': [1000], 'seed_offset': [24002000000], },
                'd+5.00|g0.02': {'seed_extent': [1000], 'seed_offset': [25002000000], },
                'd+6.00|g0.02': {'seed_extent': [1000], 'seed_offset': [26002000000], },

                'd-6.00|g0.03': {'seed_extent': [1000], 'seed_offset': [16003000000], },
                'd-5.00|g0.03': {'seed_extent': [1000], 'seed_offset': [15003000000], },
                'd-4.00|g0.03': {'seed_extent': [1000], 'seed_offset': [14003000000], },
                'd-3.00|g0.03': {'seed_extent': [1000], 'seed_offset': [13003000000], },
                'd-2.00|g0.03': {'seed_extent': [1000], 'seed_offset': [12003000000], },
                'd-1.00|g0.03': {'seed_extent': [1000], 'seed_offset': [11003000000], },
                'd+0.00|g0.03': {'seed_extent': [1000], 'seed_offset': [20003000000], },
                'd+1.00|g0.03': {'seed_extent': [1000], 'seed_offset': [21003000000], },
                'd+2.00|g0.03': {'seed_extent': [1000], 'seed_offset': [22003000000], },
                'd+3.00|g0.03': {'seed_extent': [1000], 'seed_offset': [23003000000], },
                'd+4.00|g0.03': {'seed_extent': [1000], 'seed_offset': [24003000000], },
                'd+5.00|g0.03': {'seed_extent': [1000], 'seed_offset': [25003000000], },
                'd+6.00|g0.03': {'seed_extent': [1000], 'seed_offset': [26003000000], },

            }
        },
        'xdmrg2-tuned': {
            'description': "Second batch in 2024 for xDMRG using forward subsystem expansion and deparallelization."
                           "The first batch had several types of convergence issues on different rare events that "
                           "should now be addressed."
                           "The seeds encode the Hamiltonian parameters Eg: 16|00|00000"
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The next three second digit represents the g-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg2-tuned',
            'batch': {
                'd-6.00|g0.00': {'seed_extent': [1000], 'seed_offset': [16000000000], },
                'd-5.00|g0.00': {'seed_extent': [1000], 'seed_offset': [15000000000], },
                'd-4.00|g0.00': {'seed_extent': [1000], 'seed_offset': [14000000000], },
                'd-3.00|g0.00': {'seed_extent': [1000], 'seed_offset': [13000000000], },
                'd-2.00|g0.00': {'seed_extent': [1000], 'seed_offset': [12000000000], },
                'd-1.00|g0.00': {'seed_extent': [1000], 'seed_offset': [11000000000], },
                'd+0.00|g0.00': {'seed_extent': [1000], 'seed_offset': [20000000000], },
                'd+1.00|g0.00': {'seed_extent': [1000], 'seed_offset': [21000000000], },
                'd+2.00|g0.00': {'seed_extent': [1000], 'seed_offset': [22000000000], },
                'd+3.00|g0.00': {'seed_extent': [1000], 'seed_offset': [23000000000], },
                'd+4.00|g0.00': {'seed_extent': [1000], 'seed_offset': [24000000000], },
                'd+5.00|g0.00': {'seed_extent': [1000], 'seed_offset': [25000000000], },
                'd+6.00|g0.00': {'seed_extent': [1000], 'seed_offset': [26000000000], },

                'd-6.00|g0.01': {'seed_extent': [1000], 'seed_offset': [16001000000], },
                'd-5.00|g0.01': {'seed_extent': [1000], 'seed_offset': [15001000000], },
                'd-4.00|g0.01': {'seed_extent': [1000], 'seed_offset': [14001000000], },
                'd-3.00|g0.01': {'seed_extent': [1000], 'seed_offset': [13001000000], },
                'd-2.00|g0.01': {'seed_extent': [1000], 'seed_offset': [12001000000], },
                'd-1.00|g0.01': {'seed_extent': [1000], 'seed_offset': [11001000000], },
                'd+0.00|g0.01': {'seed_extent': [1000], 'seed_offset': [20001000000], },
                'd+1.00|g0.01': {'seed_extent': [1000], 'seed_offset': [21001000000], },
                'd+2.00|g0.01': {'seed_extent': [1000], 'seed_offset': [22001000000], },
                'd+3.00|g0.01': {'seed_extent': [1000], 'seed_offset': [23001000000], },
                'd+4.00|g0.01': {'seed_extent': [1000], 'seed_offset': [24001000000], },
                'd+5.00|g0.01': {'seed_extent': [1000], 'seed_offset': [25001000000], },
                'd+6.00|g0.01': {'seed_extent': [1000], 'seed_offset': [26001000000], },

                'd-6.00|g0.02': {'seed_extent': [1000], 'seed_offset': [16002000000], },
                'd-5.00|g0.02': {'seed_extent': [1000], 'seed_offset': [15002000000], },
                'd-4.00|g0.02': {'seed_extent': [1000], 'seed_offset': [14002000000], },
                'd-3.00|g0.02': {'seed_extent': [1000], 'seed_offset': [13002000000], },
                'd-2.00|g0.02': {'seed_extent': [1000], 'seed_offset': [12002000000], },
                'd-1.00|g0.02': {'seed_extent': [1000], 'seed_offset': [11002000000], },
                'd+0.00|g0.02': {'seed_extent': [1000], 'seed_offset': [20002000000], },
                'd+1.00|g0.02': {'seed_extent': [1000], 'seed_offset': [21002000000], },
                'd+2.00|g0.02': {'seed_extent': [1000], 'seed_offset': [22002000000], },
                'd+3.00|g0.02': {'seed_extent': [1000], 'seed_offset': [23002000000], },
                'd+4.00|g0.02': {'seed_extent': [1000], 'seed_offset': [24002000000], },
                'd+5.00|g0.02': {'seed_extent': [1000], 'seed_offset': [25002000000], },
                'd+6.00|g0.02': {'seed_extent': [1000], 'seed_offset': [26002000000], },

                'd-6.00|g0.03': {'seed_extent': [1000], 'seed_offset': [16003000000], },
                'd-5.00|g0.03': {'seed_extent': [1000], 'seed_offset': [15003000000], },
                'd-4.00|g0.03': {'seed_extent': [1000], 'seed_offset': [14003000000], },
                'd-3.00|g0.03': {'seed_extent': [1000], 'seed_offset': [13003000000], },
                'd-2.00|g0.03': {'seed_extent': [1000], 'seed_offset': [12003000000], },
                'd-1.00|g0.03': {'seed_extent': [1000], 'seed_offset': [11003000000], },
                'd+0.00|g0.03': {'seed_extent': [1000], 'seed_offset': [20003000000], },
                'd+1.00|g0.03': {'seed_extent': [1000], 'seed_offset': [21003000000], },
                'd+2.00|g0.03': {'seed_extent': [1000], 'seed_offset': [22003000000], },
                'd+3.00|g0.03': {'seed_extent': [1000], 'seed_offset': [23003000000], },
                'd+4.00|g0.03': {'seed_extent': [1000], 'seed_offset': [24003000000], },
                'd+5.00|g0.03': {'seed_extent': [1000], 'seed_offset': [25003000000], },
                'd+6.00|g0.03': {'seed_extent': [1000], 'seed_offset': [26003000000], },

            }
        },
        'xdmrg3-letsgo': {
            'description': "Third batch in 2024 for xDMRG using forward subsystem expansion and deparallelization."
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg3-letsgo',
            'batch': {
                'd-6.00|g0.000': {'seed_extent': [1000], 'seed_offset': [1_600_0000_000000], },
                'd-5.00|g0.000': {'seed_extent': [1000], 'seed_offset': [1_500_0000_000000], },
                'd-4.00|g0.000': {'seed_extent': [1000], 'seed_offset': [1_400_0000_000000], },
                'd-3.00|g0.000': {'seed_extent': [1000], 'seed_offset': [1_300_0000_000000], },
                'd-2.00|g0.000': {'seed_extent': [1000], 'seed_offset': [1_200_0000_000000], },
                'd-1.50|g0.000': {'seed_extent': [1000], 'seed_offset': [1_150_0000_000000], },
                'd-1.25|g0.000': {'seed_extent': [1000], 'seed_offset': [1_125_0000_000000], },
                'd-1.00|g0.000': {'seed_extent': [1000], 'seed_offset': [1_100_0000_000000], },
                'd-0.75|g0.000': {'seed_extent': [1000], 'seed_offset': [1_075_0000_000000], },
                'd-0.50|g0.000': {'seed_extent': [1000], 'seed_offset': [1_050_0000_000000], },
                'd-0.25|g0.000': {'seed_extent': [1000], 'seed_offset': [1_025_0000_000000], },
                'd+0.00|g0.000': {'seed_extent': [1000], 'seed_offset': [2_000_0000_000000], },
                'd+0.25|g0.000': {'seed_extent': [1000], 'seed_offset': [2_025_0000_000000], },
                'd+0.50|g0.000': {'seed_extent': [1000], 'seed_offset': [2_050_0000_000000], },
                'd+0.75|g0.000': {'seed_extent': [1000], 'seed_offset': [2_075_0000_000000], },
                'd+1.00|g0.000': {'seed_extent': [1000], 'seed_offset': [2_100_0000_000000], },
                'd+1.25|g0.000': {'seed_extent': [1000], 'seed_offset': [2_125_0000_000000], },
                'd+1.50|g0.000': {'seed_extent': [1000], 'seed_offset': [2_150_0000_000000], },
                'd+2.00|g0.000': {'seed_extent': [1000], 'seed_offset': [2_200_0000_000000], },
                'd+3.00|g0.000': {'seed_extent': [1000], 'seed_offset': [2_300_0000_000000], },
                'd+4.00|g0.000': {'seed_extent': [1000], 'seed_offset': [2_400_0000_000000], },
                'd+5.00|g0.000': {'seed_extent': [1000], 'seed_offset': [2_500_0000_000000], },
                'd+6.00|g0.000': {'seed_extent': [1000], 'seed_offset': [2_600_0000_000000], },

                'd-6.00|g0.005': {'seed_extent': [1000], 'seed_offset': [1_600_0005_000000], },
                'd-5.00|g0.005': {'seed_extent': [1000], 'seed_offset': [1_500_0005_000000], },
                'd-4.00|g0.005': {'seed_extent': [1000], 'seed_offset': [1_400_0005_000000], },
                'd-3.00|g0.005': {'seed_extent': [1000], 'seed_offset': [1_300_0005_000000], },
                'd-2.00|g0.005': {'seed_extent': [1000], 'seed_offset': [1_200_0005_000000], },
                'd-1.00|g0.005': {'seed_extent': [1000], 'seed_offset': [1_100_0005_000000], },
                'd-0.75|g0.005': {'seed_extent': [1000], 'seed_offset': [1_075_0005_000000], },
                'd-0.50|g0.005': {'seed_extent': [1000], 'seed_offset': [1_050_0005_000000], },
                'd-0.25|g0.005': {'seed_extent': [1000], 'seed_offset': [1_025_0005_000000], },
                'd+0.00|g0.005': {'seed_extent': [1000], 'seed_offset': [2_000_0005_000000], },
                'd+0.25|g0.005': {'seed_extent': [1000], 'seed_offset': [2_025_0005_000000], },
                'd+0.50|g0.005': {'seed_extent': [1000], 'seed_offset': [2_050_0005_000000], },
                'd+0.75|g0.005': {'seed_extent': [1000], 'seed_offset': [2_075_0005_000000], },
                'd+1.00|g0.005': {'seed_extent': [1000], 'seed_offset': [2_100_0005_000000], },
                'd+2.00|g0.005': {'seed_extent': [1000], 'seed_offset': [2_200_0005_000000], },
                'd+3.00|g0.005': {'seed_extent': [1000], 'seed_offset': [2_300_0005_000000], },
                'd+4.00|g0.005': {'seed_extent': [1000], 'seed_offset': [2_400_0005_000000], },
                'd+5.00|g0.005': {'seed_extent': [1000], 'seed_offset': [2_500_0005_000000], },
                'd+6.00|g0.005': {'seed_extent': [1000], 'seed_offset': [2_600_0005_000000], },

                'd-6.00|g0.010': {'seed_extent': [1000], 'seed_offset': [1_600_0010_000000], },
                'd-5.00|g0.010': {'seed_extent': [1000], 'seed_offset': [1_500_0010_000000], },
                'd-4.00|g0.010': {'seed_extent': [1000], 'seed_offset': [1_400_0010_000000], },
                'd-3.00|g0.010': {'seed_extent': [1000], 'seed_offset': [1_300_0010_000000], },
                'd-2.00|g0.010': {'seed_extent': [1000], 'seed_offset': [1_200_0010_000000], },
                'd-1.00|g0.010': {'seed_extent': [1000], 'seed_offset': [1_100_0010_000000], },
                'd-0.75|g0.010': {'seed_extent': [1000], 'seed_offset': [1_075_0010_000000], },
                'd-0.50|g0.010': {'seed_extent': [1000], 'seed_offset': [1_050_0010_000000], },
                'd-0.25|g0.010': {'seed_extent': [1000], 'seed_offset': [1_025_0010_000000], },
                'd+0.00|g0.010': {'seed_extent': [1000], 'seed_offset': [2_000_0010_000000], },
                'd+0.25|g0.010': {'seed_extent': [1000], 'seed_offset': [2_025_0010_000000], },
                'd+0.50|g0.010': {'seed_extent': [1000], 'seed_offset': [2_050_0010_000000], },
                'd+0.75|g0.010': {'seed_extent': [1000], 'seed_offset': [2_075_0010_000000], },
                'd+1.00|g0.010': {'seed_extent': [1000], 'seed_offset': [2_100_0010_000000], },
                'd+2.00|g0.010': {'seed_extent': [1000], 'seed_offset': [2_200_0010_000000], },
                'd+3.00|g0.010': {'seed_extent': [1000], 'seed_offset': [2_300_0010_000000], },
                'd+4.00|g0.010': {'seed_extent': [1000], 'seed_offset': [2_400_0010_000000], },
                'd+5.00|g0.010': {'seed_extent': [1000], 'seed_offset': [2_500_0010_000000], },
                'd+6.00|g0.010': {'seed_extent': [1000], 'seed_offset': [2_600_0010_000000], },

                'd-6.00|g0.015': {'seed_extent': [1000], 'seed_offset': [1_600_0015_000000], },
                'd-5.00|g0.015': {'seed_extent': [1000], 'seed_offset': [1_500_0015_000000], },
                'd-4.00|g0.015': {'seed_extent': [1000], 'seed_offset': [1_400_0015_000000], },
                'd-3.00|g0.015': {'seed_extent': [1000], 'seed_offset': [1_300_0015_000000], },
                'd-2.00|g0.015': {'seed_extent': [1000], 'seed_offset': [1_200_0015_000000], },
                'd-1.00|g0.015': {'seed_extent': [1000], 'seed_offset': [1_100_0015_000000], },
                'd-0.75|g0.015': {'seed_extent': [1000], 'seed_offset': [1_075_0015_000000], },
                'd-0.50|g0.015': {'seed_extent': [1000], 'seed_offset': [1_050_0015_000000], },
                'd-0.25|g0.015': {'seed_extent': [1000], 'seed_offset': [1_025_0015_000000], },
                'd+0.00|g0.015': {'seed_extent': [1000], 'seed_offset': [2_000_0015_000000], },
                'd+0.25|g0.015': {'seed_extent': [1000], 'seed_offset': [2_025_0015_000000], },
                'd+0.50|g0.015': {'seed_extent': [1000], 'seed_offset': [2_050_0015_000000], },
                'd+0.75|g0.015': {'seed_extent': [1000], 'seed_offset': [2_075_0015_000000], },
                'd+1.00|g0.015': {'seed_extent': [1000], 'seed_offset': [2_100_0015_000000], },
                'd+2.00|g0.015': {'seed_extent': [1000], 'seed_offset': [2_200_0015_000000], },
                'd+3.00|g0.015': {'seed_extent': [1000], 'seed_offset': [2_300_0015_000000], },
                'd+4.00|g0.015': {'seed_extent': [1000], 'seed_offset': [2_400_0015_000000], },
                'd+5.00|g0.015': {'seed_extent': [1000], 'seed_offset': [2_500_0015_000000], },
                'd+6.00|g0.015': {'seed_extent': [1000], 'seed_offset': [2_600_0015_000000], },

                'd-6.00|g0.020': {'seed_extent': [1000], 'seed_offset': [1_600_0020_000000], },
                'd-5.00|g0.020': {'seed_extent': [1000], 'seed_offset': [1_500_0020_000000], },
                'd-4.00|g0.020': {'seed_extent': [1000], 'seed_offset': [1_400_0020_000000], },
                'd-3.00|g0.020': {'seed_extent': [1000], 'seed_offset': [1_300_0020_000000], },
                'd-2.00|g0.020': {'seed_extent': [1000], 'seed_offset': [1_200_0020_000000], },
                'd-1.00|g0.020': {'seed_extent': [1000], 'seed_offset': [1_100_0020_000000], },
                'd-0.75|g0.020': {'seed_extent': [1000], 'seed_offset': [1_075_0020_000000], },
                'd-0.50|g0.020': {'seed_extent': [1000], 'seed_offset': [1_050_0020_000000], },
                'd-0.25|g0.020': {'seed_extent': [1000], 'seed_offset': [1_025_0020_000000], },
                'd+0.00|g0.020': {'seed_extent': [1000], 'seed_offset': [2_000_0020_000000], },
                'd+0.25|g0.020': {'seed_extent': [1000], 'seed_offset': [2_025_0020_000000], },
                'd+0.50|g0.020': {'seed_extent': [1000], 'seed_offset': [2_050_0020_000000], },
                'd+0.75|g0.020': {'seed_extent': [1000], 'seed_offset': [2_075_0020_000000], },
                'd+1.00|g0.020': {'seed_extent': [1000], 'seed_offset': [2_100_0020_000000], },
                'd+2.00|g0.020': {'seed_extent': [1000], 'seed_offset': [2_200_0020_000000], },
                'd+3.00|g0.020': {'seed_extent': [1000], 'seed_offset': [2_300_0020_000000], },
                'd+4.00|g0.020': {'seed_extent': [1000], 'seed_offset': [2_400_0020_000000], },
                'd+5.00|g0.020': {'seed_extent': [1000], 'seed_offset': [2_500_0020_000000], },
                'd+6.00|g0.020': {'seed_extent': [1000], 'seed_offset': [2_600_0020_000000], },

                'd-6.00|g0.025': {'seed_extent': [1000], 'seed_offset': [1_600_0025_000000], },
                'd-5.00|g0.025': {'seed_extent': [1000], 'seed_offset': [1_500_0025_000000], },
                'd-4.00|g0.025': {'seed_extent': [1000], 'seed_offset': [1_400_0025_000000], },
                'd-3.00|g0.025': {'seed_extent': [1000], 'seed_offset': [1_300_0025_000000], },
                'd-2.00|g0.025': {'seed_extent': [1000], 'seed_offset': [1_200_0025_000000], },
                'd-1.00|g0.025': {'seed_extent': [1000], 'seed_offset': [1_100_0025_000000], },
                'd-0.75|g0.025': {'seed_extent': [1000], 'seed_offset': [1_075_0025_000000], },
                'd-0.50|g0.025': {'seed_extent': [1000], 'seed_offset': [1_050_0025_000000], },
                'd-0.25|g0.025': {'seed_extent': [1000], 'seed_offset': [1_025_0025_000000], },
                'd+0.00|g0.025': {'seed_extent': [1000], 'seed_offset': [2_000_0025_000000], },
                'd+0.25|g0.025': {'seed_extent': [1000], 'seed_offset': [2_025_0025_000000], },
                'd+0.50|g0.025': {'seed_extent': [1000], 'seed_offset': [2_050_0025_000000], },
                'd+0.75|g0.025': {'seed_extent': [1000], 'seed_offset': [2_075_0025_000000], },
                'd+1.00|g0.025': {'seed_extent': [1000], 'seed_offset': [2_100_0025_000000], },
                'd+2.00|g0.025': {'seed_extent': [1000], 'seed_offset': [2_200_0025_000000], },
                'd+3.00|g0.025': {'seed_extent': [1000], 'seed_offset': [2_300_0025_000000], },
                'd+4.00|g0.025': {'seed_extent': [1000], 'seed_offset': [2_400_0025_000000], },
                'd+5.00|g0.025': {'seed_extent': [1000], 'seed_offset': [2_500_0025_000000], },
                'd+6.00|g0.025': {'seed_extent': [1000], 'seed_offset': [2_600_0025_000000], },

                'd-6.00|g0.030': {'seed_extent': [1000], 'seed_offset': [1_600_0030_000000], },
                'd-5.00|g0.030': {'seed_extent': [1000], 'seed_offset': [1_500_0030_000000], },
                'd-4.00|g0.030': {'seed_extent': [1000], 'seed_offset': [1_400_0030_000000], },
                'd-3.00|g0.030': {'seed_extent': [1000], 'seed_offset': [1_300_0030_000000], },
                'd-2.00|g0.030': {'seed_extent': [1000], 'seed_offset': [1_200_0030_000000], },
                'd-1.00|g0.030': {'seed_extent': [1000], 'seed_offset': [1_100_0030_000000], },
                'd-0.75|g0.030': {'seed_extent': [1000], 'seed_offset': [1_075_0030_000000], },
                'd-0.50|g0.030': {'seed_extent': [1000], 'seed_offset': [1_050_0030_000000], },
                'd-0.25|g0.030': {'seed_extent': [1000], 'seed_offset': [1_025_0030_000000], },
                'd+0.00|g0.030': {'seed_extent': [1000], 'seed_offset': [2_000_0030_000000], },
                'd+0.25|g0.030': {'seed_extent': [1000], 'seed_offset': [2_025_0030_000000], },
                'd+0.50|g0.030': {'seed_extent': [1000], 'seed_offset': [2_050_0030_000000], },
                'd+0.75|g0.030': {'seed_extent': [1000], 'seed_offset': [2_075_0030_000000], },
                'd+1.00|g0.030': {'seed_extent': [1000], 'seed_offset': [2_100_0030_000000], },
                'd+2.00|g0.030': {'seed_extent': [1000], 'seed_offset': [2_200_0030_000000], },
                'd+3.00|g0.030': {'seed_extent': [1000], 'seed_offset': [2_300_0030_000000], },
                'd+4.00|g0.030': {'seed_extent': [1000], 'seed_offset': [2_400_0030_000000], },
                'd+5.00|g0.030': {'seed_extent': [1000], 'seed_offset': [2_500_0030_000000], },
                'd+6.00|g0.030': {'seed_extent': [1000], 'seed_offset': [2_600_0030_000000], },

                'd-6.00|g0.040': {'seed_extent': [1000], 'seed_offset': [1_600_0040_000000], },
                'd-5.00|g0.040': {'seed_extent': [1000], 'seed_offset': [1_500_0040_000000], },
                'd-4.00|g0.040': {'seed_extent': [1000], 'seed_offset': [1_400_0040_000000], },
                'd-3.00|g0.040': {'seed_extent': [1000], 'seed_offset': [1_300_0040_000000], },
                'd-2.00|g0.040': {'seed_extent': [1000], 'seed_offset': [1_200_0040_000000], },
                'd-1.00|g0.040': {'seed_extent': [1000], 'seed_offset': [1_100_0040_000000], },
                'd-0.75|g0.040': {'seed_extent': [1000], 'seed_offset': [1_075_0040_000000], },
                'd-0.50|g0.040': {'seed_extent': [1000], 'seed_offset': [1_050_0040_000000], },
                'd-0.25|g0.040': {'seed_extent': [1000], 'seed_offset': [1_025_0040_000000], },
                'd+0.00|g0.040': {'seed_extent': [1000], 'seed_offset': [2_000_0040_000000], },
                'd+0.25|g0.040': {'seed_extent': [1000], 'seed_offset': [2_025_0040_000000], },
                'd+0.50|g0.040': {'seed_extent': [1000], 'seed_offset': [2_050_0040_000000], },
                'd+0.75|g0.040': {'seed_extent': [1000], 'seed_offset': [2_075_0040_000000], },
                'd+1.00|g0.040': {'seed_extent': [1000], 'seed_offset': [2_100_0040_000000], },
                'd+2.00|g0.040': {'seed_extent': [1000], 'seed_offset': [2_200_0040_000000], },
                'd+3.00|g0.040': {'seed_extent': [1000], 'seed_offset': [2_300_0040_000000], },
                'd+4.00|g0.040': {'seed_extent': [1000], 'seed_offset': [2_400_0040_000000], },
                'd+5.00|g0.040': {'seed_extent': [1000], 'seed_offset': [2_500_0040_000000], },
                'd+6.00|g0.040': {'seed_extent': [1000], 'seed_offset': [2_600_0040_000000], },

                'd-6.00|g0.060': {'seed_extent': [1000], 'seed_offset': [1_600_0060_000000], },
                'd-5.00|g0.060': {'seed_extent': [1000], 'seed_offset': [1_500_0060_000000], },
                'd-4.00|g0.060': {'seed_extent': [1000], 'seed_offset': [1_400_0060_000000], },
                'd-3.00|g0.060': {'seed_extent': [1000], 'seed_offset': [1_300_0060_000000], },
                'd-2.00|g0.060': {'seed_extent': [1000], 'seed_offset': [1_200_0060_000000], },
                'd-1.00|g0.060': {'seed_extent': [1000], 'seed_offset': [1_100_0060_000000], },
                'd-0.75|g0.060': {'seed_extent': [1000], 'seed_offset': [1_075_0060_000000], },
                'd-0.50|g0.060': {'seed_extent': [1000], 'seed_offset': [1_050_0060_000000], },
                'd-0.25|g0.060': {'seed_extent': [1000], 'seed_offset': [1_025_0060_000000], },
                'd+0.00|g0.060': {'seed_extent': [1000], 'seed_offset': [2_000_0060_000000], },
                'd+0.25|g0.060': {'seed_extent': [1000], 'seed_offset': [2_025_0060_000000], },
                'd+0.50|g0.060': {'seed_extent': [1000], 'seed_offset': [2_050_0060_000000], },
                'd+0.75|g0.060': {'seed_extent': [1000], 'seed_offset': [2_075_0060_000000], },
                'd+1.00|g0.060': {'seed_extent': [1000], 'seed_offset': [2_100_0060_000000], },
                'd+2.00|g0.060': {'seed_extent': [1000], 'seed_offset': [2_200_0060_000000], },
                'd+3.00|g0.060': {'seed_extent': [1000], 'seed_offset': [2_300_0060_000000], },
                'd+4.00|g0.060': {'seed_extent': [1000], 'seed_offset': [2_400_0060_000000], },
                'd+5.00|g0.060': {'seed_extent': [1000], 'seed_offset': [2_500_0060_000000], },
                'd+6.00|g0.060': {'seed_extent': [1000], 'seed_offset': [2_600_0060_000000], },

                'd-6.00|g0.080': {'seed_extent': [1000], 'seed_offset': [1_600_0080_000000], },
                'd-5.00|g0.080': {'seed_extent': [1000], 'seed_offset': [1_500_0080_000000], },
                'd-4.00|g0.080': {'seed_extent': [1000], 'seed_offset': [1_400_0080_000000], },
                'd-3.00|g0.080': {'seed_extent': [1000], 'seed_offset': [1_300_0080_000000], },
                'd-2.00|g0.080': {'seed_extent': [1000], 'seed_offset': [1_200_0080_000000], },
                'd-1.00|g0.080': {'seed_extent': [1000], 'seed_offset': [1_100_0080_000000], },
                'd-0.75|g0.080': {'seed_extent': [1000], 'seed_offset': [1_075_0080_000000], },
                'd-0.50|g0.080': {'seed_extent': [1000], 'seed_offset': [1_050_0080_000000], },
                'd-0.25|g0.080': {'seed_extent': [1000], 'seed_offset': [1_025_0080_000000], },
                'd+0.00|g0.080': {'seed_extent': [1000], 'seed_offset': [2_000_0080_000000], },
                'd+0.25|g0.080': {'seed_extent': [1000], 'seed_offset': [2_025_0080_000000], },
                'd+0.50|g0.080': {'seed_extent': [1000], 'seed_offset': [2_050_0080_000000], },
                'd+0.75|g0.080': {'seed_extent': [1000], 'seed_offset': [2_075_0080_000000], },
                'd+1.00|g0.080': {'seed_extent': [1000], 'seed_offset': [2_100_0080_000000], },
                'd+2.00|g0.080': {'seed_extent': [1000], 'seed_offset': [2_200_0080_000000], },
                'd+3.00|g0.080': {'seed_extent': [1000], 'seed_offset': [2_300_0080_000000], },
                'd+4.00|g0.080': {'seed_extent': [1000], 'seed_offset': [2_400_0080_000000], },
                'd+5.00|g0.080': {'seed_extent': [1000], 'seed_offset': [2_500_0080_000000], },
                'd+6.00|g0.080': {'seed_extent': [1000], 'seed_offset': [2_600_0080_000000], },

                'd-6.00|g0.100': {'seed_extent': [1000], 'seed_offset': [1_600_0100_000000], },
                'd-5.00|g0.100': {'seed_extent': [1000], 'seed_offset': [1_500_0100_000000], },
                'd-4.00|g0.100': {'seed_extent': [1000], 'seed_offset': [1_400_0100_000000], },
                'd-3.00|g0.100': {'seed_extent': [1000], 'seed_offset': [1_300_0100_000000], },
                'd-2.00|g0.100': {'seed_extent': [1000], 'seed_offset': [1_200_0100_000000], },
                'd-1.00|g0.100': {'seed_extent': [1000], 'seed_offset': [1_100_0100_000000], },
                'd-0.75|g0.100': {'seed_extent': [1000], 'seed_offset': [1_075_0100_000000], },
                'd-0.50|g0.100': {'seed_extent': [1000], 'seed_offset': [1_050_0100_000000], },
                'd-0.25|g0.100': {'seed_extent': [1000], 'seed_offset': [1_025_0100_000000], },
                'd+0.00|g0.100': {'seed_extent': [1000], 'seed_offset': [2_000_0100_000000], },
                'd+0.25|g0.100': {'seed_extent': [1000], 'seed_offset': [2_025_0100_000000], },
                'd+0.50|g0.100': {'seed_extent': [1000], 'seed_offset': [2_050_0100_000000], },
                'd+0.75|g0.100': {'seed_extent': [1000], 'seed_offset': [2_075_0100_000000], },
                'd+1.00|g0.100': {'seed_extent': [1000], 'seed_offset': [2_100_0100_000000], },
                'd+2.00|g0.100': {'seed_extent': [1000], 'seed_offset': [2_200_0100_000000], },
                'd+3.00|g0.100': {'seed_extent': [1000], 'seed_offset': [2_300_0100_000000], },
                'd+4.00|g0.100': {'seed_extent': [1000], 'seed_offset': [2_400_0100_000000], },
                'd+5.00|g0.100': {'seed_extent': [1000], 'seed_offset': [2_500_0100_000000], },
                'd+6.00|g0.100': {'seed_extent': [1000], 'seed_offset': [2_600_0100_000000], },

            }
        },

        'xdmrg4-exp-1site-back': {
            'description': "xDMRG using 1-site backward expansion."
                           "Forward, backward, 1site, nsite (n==icom)"
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg4-exp-1site-back',
            'batch': {
                'L16|d+0.50|g0.100': {'seed_extent': [500], 'seed_offset': [2_050_0100_000000], },
                'L16|d+2.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_200_0100_000000], },
                'L20|d-4.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_400_0100_000000], },
                'L20|d-3.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_300_0100_000000], },
                'L20|d+3.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_300_0100_000000], },
                'L20|d+4.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_400_0100_000000], },
            }
        },
        'xdmrg4-exp-1site-forw': {
            'description': "xDMRG using 1-site forward expansion."
                           "Forward, backward, 1site, nsite (n==icom)"
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg4-exp-1site-forw',
            'batch': {
                'L16|d+0.50|g0.100': {'seed_extent': [500], 'seed_offset': [2_050_0100_000000], },
                'L16|d+2.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_200_0100_000000], },
                'L20|d-4.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_400_0100_000000], },
                'L20|d-3.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_300_0100_000000], },
                'L20|d+3.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_300_0100_000000], },
                'L20|d+4.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_400_0100_000000], },

            }
        },
        'xdmrg4-exp-2site-forw': {
            'description': "xDMRG using 2-site forward expansion."
                           "Forward, backward, 1site, nsite (n==icom)"
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg4-exp-2site-forw',
            'batch': {
                'L16|d+0.50|g0.100': {'seed_extent': [500], 'seed_offset': [2_050_0100_000000], },
                'L16|d+2.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_200_0100_000000], },
                'L20|d-4.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_400_0100_000000], },
                'L20|d-3.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_300_0100_000000], },
                'L20|d+3.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_300_0100_000000], },
                'L20|d+4.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_400_0100_000000], },
            }
        },
        'xdmrg4-exp-3site-forw': {
            'description': "xDMRG using 2-site forward expansion."
                           "Forward, backward, 1site, nsite (n==icom)"
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg4-exp-3site-forw',
            'batch': {
                'L16|d+0.50|g0.100': {'seed_extent': [500], 'seed_offset': [2_050_0100_000000], },
                'L16|d+2.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_200_0100_000000], },
                'L20|d-4.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_400_0100_000000], },
                'L20|d-3.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_300_0100_000000], },
                'L20|d+3.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_300_0100_000000], },
                'L20|d+4.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_400_0100_000000], },
            }
        },
        'xdmrg4-exp-4site-forw': {
            'description': "xDMRG using 2-site forward expansion."
                           "Forward, backward, 1site, nsite (n==icom)"
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg4-exp-4site-forw',
            'batch': {
                'L16|d+0.50|g0.100': {'seed_extent': [500], 'seed_offset': [2_050_0100_000000], },
                'L16|d+2.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_200_0100_000000], },
                'L20|d-4.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_400_0100_000000], },
                'L20|d-3.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_300_0100_000000], },
                'L20|d+3.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_300_0100_000000], },
                'L20|d+4.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_400_0100_000000], },
            }
        },
        'xdmrg4-exp-isite-forw': {
            'description': "xDMRG using n-site (up to icom) forward expansion."
                           "Forward, backward, 1site, nsite (n==icom)"
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg4-exp-isite-forw',
            'batch': {
                'L16|d+0.50|g0.100': {'seed_extent': [500], 'seed_offset': [2_050_0100_000000], },
                'L16|d+2.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_200_0100_000000], },
                'L20|d-4.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_400_0100_000000], },
                'L20|d-3.00|g0.100': {'seed_extent': [500], 'seed_offset': [1_300_0100_000000], },
                'L20|d+3.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_300_0100_000000], },
                'L20|d+4.00|g0.100': {'seed_extent': [500], 'seed_offset': [2_400_0100_000000], },
            }
        },


        'xdmrg6-gdplusk': {
            'description': "Batch in 2025 for xDMRG with the new GD+k, JD correction equation"
                           "solved with MINRES with internal block-jacobi preconditioner."
                           "We also have a new eigensolver-based environment expansion routine."
                           ""
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg6-gdplusk',
            'batch': {
                'd-8.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_800_0500_000000], },
                'd-7.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_750_0500_000000], },
                'd-7.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_700_0500_000000], },
                'd-6.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_650_0500_000000], },
                'd-6.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_600_0500_000000], },
                'd-5.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_550_0500_000000], },
                'd-5.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_500_0500_000000], },
                'd-4.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_450_0500_000000], },
                'd-4.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_400_0500_000000], },
                'd-3.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_350_0500_000000], },
                'd-3.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_300_0500_000000], },
                'd-2.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_250_0500_000000], },
                'd-2.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_200_0500_000000], },
                'd-1.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_150_0500_000000], },
                'd-1.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_100_0500_000000], },
                'd-0.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_050_0500_000000], },
                'd+0.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_000_0500_000000], },
                'd+0.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_050_0500_000000], },
                'd+1.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_100_0500_000000], },
                'd+1.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_150_0500_000000], },
                'd+2.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_200_0500_000000], },
                'd+2.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_250_0500_000000], },
                'd+3.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_300_0500_000000], },
                'd+3.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_350_0500_000000], },
                'd+4.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_400_0500_000000], },
                'd+4.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_450_0500_000000], },
                'd+5.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_500_0500_000000], },
                'd+5.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_550_0500_000000], },
                'd+6.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_600_0500_000000], },
                'd+6.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_650_0500_000000], },
                'd+7.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_700_0500_000000], },
                'd+7.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_750_0500_000000], },
                'd+8.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_800_0500_000000], },

            }
        },
        'xdmrg6-gdplusk-L18': {
            'description': "Batch in 2025 for xDMRG with the new GD+k, JD correction equation"
                           "solved with MINRES with internal block-jacobi preconditioner."
                           "We also have a new eigensolver-based environment expansion routine."
                           ""
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg6-gdplusk',
            'seed_max' : 1000, # Consider up to this many seeds when scanning folders
            'batch': {
                'd-8.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_800_0500_000000]},
                'd-7.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_750_0500_000000]},
                'd-7.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_700_0500_000000]},
                'd-6.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_650_0500_000000]},
                'd-6.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_600_0500_000000]},
                'd-5.50|g0.500': {'seed_extent': [ 500], 'seed_offset': [1_550_0500_000000]},
                'd-5.00|g0.500': {'seed_extent': [ 300], 'seed_offset': [1_500_0500_000000]},
                'd-4.50|g0.500': {'seed_extent': [ 200], 'seed_offset': [1_450_0500_000000]},
                'd-4.00|g0.500': {'seed_extent': [ 100], 'seed_offset': [1_400_0500_000000]},
                'd-3.50|g0.500': {'seed_extent': [  50], 'seed_offset': [1_350_0500_000000]},
                'd-3.00|g0.500': {'seed_extent': [  50], 'seed_offset': [1_300_0500_000000]},
                'd-2.50|g0.500': {'seed_extent': [  50], 'seed_offset': [1_250_0500_000000]},
                'd-2.00|g0.500': {'seed_extent': [  10], 'seed_offset': [1_200_0500_000000]},
                'd-1.50|g0.500': {'seed_extent': [  10], 'seed_offset': [1_150_0500_000000]},
                'd-1.00|g0.500': {'seed_extent': [  10], 'seed_offset': [1_100_0500_000000]},
                'd-0.50|g0.500': {'seed_extent': [  10], 'seed_offset': [1_050_0500_000000]},
                'd+0.00|g0.500': {'seed_extent': [  10], 'seed_offset': [2_000_0500_000000]},
                'd+0.50|g0.500': {'seed_extent': [  10], 'seed_offset': [2_050_0500_000000]},
                'd+1.00|g0.500': {'seed_extent': [  10], 'seed_offset': [2_100_0500_000000]},
                'd+1.50|g0.500': {'seed_extent': [  10], 'seed_offset': [2_150_0500_000000]},
                'd+2.00|g0.500': {'seed_extent': [  10], 'seed_offset': [2_200_0500_000000]},
                'd+2.50|g0.500': {'seed_extent': [  50], 'seed_offset': [2_250_0500_000000]},
                'd+3.00|g0.500': {'seed_extent': [  50], 'seed_offset': [2_300_0500_000000]},
                'd+3.50|g0.500': {'seed_extent': [  50], 'seed_offset': [2_350_0500_000000]},
                'd+4.00|g0.500': {'seed_extent': [ 100], 'seed_offset': [2_400_0500_000000]},
                'd+4.50|g0.500': {'seed_extent': [ 200], 'seed_offset': [2_450_0500_000000]},
                'd+5.00|g0.500': {'seed_extent': [ 300], 'seed_offset': [2_500_0500_000000]},
                'd+5.50|g0.500': {'seed_extent': [ 500], 'seed_offset': [2_550_0500_000000]},
                'd+6.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_600_0500_000000]},
                'd+6.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_650_0500_000000]},
                'd+7.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_700_0500_000000]},
                'd+7.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_750_0500_000000]},
                'd+8.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_800_0500_000000]},

            }
        },
        'xdmrg6-gdplusk-L20': {
            'description': "Batch in 2025 for xDMRG with the new GD+k, JD correction equation"
                           "solved with MINRES with internal block-jacobi preconditioner."
                           "We also have a new eigensolver-based environment expansion routine."
                           ""
                           "The seeds encode the Hamiltonian parameters"
                           " Eg:  1|600|0000|00000"
                           "     +-| d | g  | extra "
                           "The seeds start with 1 for negative delta, and 2 for positive."
                           "The second digit represents the d-value."
                           "The remaining digits reveal are the decimals of g.",
            'projectname': 'xdmrg6-gdplusk',
            'seed_max' : 1000, # Consider up to this many seeds when scanning folders
            'batch': {
                'd-8.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_800_0500_000000]},
                'd-7.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_750_0500_000000]},
                'd-7.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_700_0500_000000]},
                'd-6.50|g0.500': {'seed_extent': [1000], 'seed_offset': [1_650_0500_000000]},
                'd-6.00|g0.500': {'seed_extent': [1000], 'seed_offset': [1_600_0500_000000]},
                'd-5.50|g0.500': {'seed_extent': [ 500], 'seed_offset': [1_550_0500_000000]},
                'd-5.00|g0.500': {'seed_extent': [ 300], 'seed_offset': [1_500_0500_000000]},
                'd-4.50|g0.500': {'seed_extent': [ 200], 'seed_offset': [1_450_0500_000000]},
                'd-4.00|g0.500': {'seed_extent': [ 100], 'seed_offset': [1_400_0500_000000]},
                'd-3.50|g0.500': {'seed_extent': [  50], 'seed_offset': [1_350_0500_000000]},
                'd-3.00|g0.500': {'seed_extent': [  25], 'seed_offset': [1_300_0500_000000]},
                'd-2.50|g0.500': {'seed_extent': [  25], 'seed_offset': [1_250_0500_000000]},
                'd-2.00|g0.500': {'seed_extent': [  10], 'seed_offset': [1_200_0500_000000]},
                'd-1.50|g0.500': {'seed_extent': [  10], 'seed_offset': [1_150_0500_000000]},
                'd-1.00|g0.500': {'seed_extent': [  10], 'seed_offset': [1_100_0500_000000]},
                'd-0.50|g0.500': {'seed_extent': [  10], 'seed_offset': [1_050_0500_000000]},
                'd+0.00|g0.500': {'seed_extent': [  10], 'seed_offset': [2_000_0500_000000]},
                'd+0.50|g0.500': {'seed_extent': [  10], 'seed_offset': [2_050_0500_000000]},
                'd+1.00|g0.500': {'seed_extent': [  10], 'seed_offset': [2_100_0500_000000]},
                'd+1.50|g0.500': {'seed_extent': [  10], 'seed_offset': [2_150_0500_000000]},
                'd+2.00|g0.500': {'seed_extent': [  10], 'seed_offset': [2_200_0500_000000]},
                'd+2.50|g0.500': {'seed_extent': [  25], 'seed_offset': [2_250_0500_000000]},
                'd+3.00|g0.500': {'seed_extent': [  25], 'seed_offset': [2_300_0500_000000]},
                'd+3.50|g0.500': {'seed_extent': [  50], 'seed_offset': [2_350_0500_000000]},
                'd+4.00|g0.500': {'seed_extent': [ 100], 'seed_offset': [2_400_0500_000000]},
                'd+4.50|g0.500': {'seed_extent': [ 200], 'seed_offset': [2_450_0500_000000]},
                'd+5.00|g0.500': {'seed_extent': [ 300], 'seed_offset': [2_500_0500_000000]},
                'd+5.50|g0.500': {'seed_extent': [ 500], 'seed_offset': [2_550_0500_000000]},
                'd+6.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_600_0500_000000]},
                'd+6.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_650_0500_000000]},
                'd+7.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_700_0500_000000]},
                'd+7.50|g0.500': {'seed_extent': [1000], 'seed_offset': [2_750_0500_000000]},
                'd+8.00|g0.500': {'seed_extent': [1000], 'seed_offset': [2_800_0500_000000]},

            }
        },



    }

    return batch_setup[projectname]
