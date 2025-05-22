__global__ void gemm_tiled(float *A, float *B, float *C, int M, int N, int K) {
    // 分配共享内存 此时在block中开辟了2个16x16大小的共享内存。
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
  //全局内存的二维转一维，A[x][y]二维 = A[y* row(width) +x]
  // 这里出现了threadidx，虽然没有循环，但是实际上是不同的thread同时执行的，
  //根据id的位置执行不同的操作 threadid x,y 属于[0，16]
  // 假定blockIdx.y = 1, blcokidx.x = 2 threaidx.y = 3, threaidx.x = 4  

    // 线程索引
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  // row = 1*16+3 = 19  col = 2 * 16 +4 =36   
    float sum = 0.0f;
    
    // 沿K维度分块
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
// t = 0 ,As[3][4] = A[row][t*tile+th.x] = A[19][4] 
//    当前的线程，这里只是加载了As[3][4]

        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
// t = 0 ,Bs[3][4] = B[t*tile+th.y][col] = B[3][36] 
    //当前的线程，这里只是加载了 Bs[3][36]

        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // 同步，确保子块加载完成
        __syncthreads();

      //!// 注意这个同步非常关键，因为如果不同步的话，只是加载了分块中的As[3][4],Bs[3][36]，后续不能构成矩阵的计算。
      //但是同步之后，此时同一个block中的有256线程全部加载完
    As[threadIdx.y][threadIdx.x] //表示每个线程根据自己的ID，将数据搬入对应的位置，
    //比如thred(1,0)这个线程搬运A[17][0]到As[1][0]，thred(1,1)这个线程搬运A[17][1]到As[1][0]
    As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    // 表示 thred(0:15,0:15)搬运A[16:31][0:15]到As[0:15][0:15] Bs同理，
    // 类似block共享内存中有16x16网格，每个线程搬运数据到自己特定的网格中。正好填充完整
    // 计算C[id.x(2)][id.y(3)] 矩阵乘法需要其他网格的值，这256个线程又在同一个block中，所以可以彼此访问共享内存，
    // 具体例子如下，同步之后，要将As[0:15][0:15] Bs[0:15][0:15]全部加载完成。
t=0,As[th.y][th.x]=A [bl.y*tile+th.y] [t*tile+th.x]
    As[0][0]=A[16][0] As[0][1]=A[16][1]
    As[1][0]=A[17][0] As[1][1]=A[17][1]
    Ax[bl.y][t] = Ax[0][1] (子块) -> As[0:15][0:15] = A[16:31][0:15]

    Bs[th.y][th.x]=A [t*tile+th.y] [bl.x*tile+th.x]
    Bs[0][0]=B[0][16] Bs[0][1]=B[0][17]
    Bs[1][0]=B[1][16] Bs[1][1]=B[1][17]
    Bx[t][bl.x] = Bx[0][1] (子块) -> Bs[0:15][0:15] = B[0:15][16:31]

      // 计算子块的乘法
      for (int k = 0; k < TILE_SIZE; k++) {
          sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
      }
      // 以上同步完之后As[0:15][0:15] Bs[0:15][0:15]中都有数据了，现在计算
      As[3][0]*Bs[0][4] + As[3][1]*Bs[1][4] + As[3][2]*Bs[2][4] ...
      sum k=0,16 (As[3][k]*Bs[k][4]) 可以得出分块矩阵乘法Cs[3][4]的值

      //如果不同步的话，那么仅仅计算当前线程Cs[3][4]的值，同步之后计算所有Cs[0:16][0:16]的值
    //   并且每一个线程都有各自的寄存器sum。供下一次累加。此时已经计算完成t=0的分块，
    //   一个小分块的16x16的分块矩阵结果。（不仅仅一个值，而是一个小分块的阶段性16x16结果。
      __syncthreads();
    }
    
    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}