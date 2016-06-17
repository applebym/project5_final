from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
import sys

def scatter(x):
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.savefig('clusters_tsne-generated.png', dpi=120)


def main():
    category = sys.argv[1]
    with open('pickle/' + category + '_keypoints_df.pickle', 'rb') as handle:
       keypoints_df = pickle.load(handle)
    print len(keypoints_df)
    sample = keypoints_df.sample(n = 10000)
    keypoints_proj = TSNE().fit_transform(sample)
    scatter(keypoints_proj)


if __name__ == '__main__':
	main()

