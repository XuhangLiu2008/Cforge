#include <torch/torch.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

using namespace std;

/*
MARK: KEY PRINCIPLE

----AIR--->] [<--Fila1-->] [<--Fila2-->] [<--........-->] [<Fila(n-1)>] [<---AIR----
            |             |             |                |             |
       E_f0 |        E_f1 |        E_f2 |                |    E_f(n-1) |        E_fn
----------> | ----------> | ----------> | ----......---> | ----------> | ---------->
            |             |             |                |             |
<---------- | <---------- | <---------- | <---......---- | <---------- | <----------
E_b0        | E_b1        | E_b2        |                | E_b(n-1)    | E_bn
            |             |             |                |             |


E : the expectation of the number of passing for one photon

E_f0, E_bn given
E_b0, E_fn needed (exactly the same as the ratio of the intensity)

modeled by Markov chain, we get:

E_fi = P[i-1][i] * E_f(i-1) + R[i][i-1] * E_bi
E_bi = P[i+1][i] * E_b(i+1) + R[i][i+1] * E_fi

for each i


where:

r[i][j] = ( (n_i - n_j) / (n_i + n_j) ) ** 2  # reflected ratio at the surface
P[i][j] = (1 - r[i][j]) * exp(-K[j] * d)
R[i][j] = r[i][j] * exp(-K[i] * d)


get E_b0 and E_fn by solving the simultaneous equations

*/

// void print(const torch::Tensor &t) {
//     cout << t << endl;
// }



/*
# 删除构建目录
rm -rf cmake-build-debug

# 重新生成
mkdir -p cmake-build-debug
cd cmake-build-debug
cmake ..
make
*/


/*
refra_index = [9.47557073 4.7540685  4.45918725]
extinc_coeff = [0.23565427 1.77511957 6.07312633]
*/
