from beetl.task_datasets import BeetlMILeaderboard

_, _, X_MIA_test = BeetlMILeaderboard().get_data(dataset='A')
print ("MI leaderboard A: There are {} trials with {} electrodes and {} time samples".format(*X_MIA_test.shape))

_, _, X_MIB_test = BeetlMILeaderboard().get_data(dataset='B')
print ("MI leaderboard B: There are {} trials with {} electrodes and {} time samples".format(*X_MIB_test.shape))
