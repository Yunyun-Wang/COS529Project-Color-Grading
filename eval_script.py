from subprocess import call
import matplotlib.pyplot as plt
import pickle
import numpy as np

points = [10, 32, 64]
result_dir = 'results'

def gen_results():
    for step in range(500, 550, 50):
        for point in points:
            cmd = 'python train.py --only_test True --load_from no_ref_{}points/step-{}.ckpt --lut_points {}'.format(point, step, point)
            print(cmd)
            call(cmd, shell=True)
            cmd = 'python train.py --only_test True --load_from give_ref_{}points/step-{}.ckpt --lut_points {} --give_ref True'.format(point, step, point)
            print(cmd)
            call(cmd, shell=True)

def comp_plot():
    results = {}
    x_axis = []
    for step in range(500, 550, 50):
        x_axis.append(step)
        for point in points:
            for give_ref in [False]:
                filename = '{}/give_ref_{}-lut_points_{}-step_{}.pkl'.format(result_dir, give_ref, point, step)
                curve_name = 'give_ref_{}-lut_points_{}'.format(give_ref, point)
                result, org_result = pickle.load(open(filename, 'rb'))
                to_add = np.mean(np.array(org_result[:30]))
                if curve_name in results:
                    results[curve_name].append(to_add)
                else:
                    results[curve_name] = [to_add]
    for curve_name in results:
        plt.plot(x_axis, results[curve_name], label=curve_name)
        print(curve_name, results[curve_name][-1])
    plt.legend()
    plt.savefig('result.png')
    plt.close()

if __name__ == '__main__':
    # gen_results()
    comp_plot()