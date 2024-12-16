using SixLabors.ImageSharp;  // 使用 ImageSharp 替代 System.Drawing
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Tensorflow;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;  // 這會引入 tf

namespace MNIST_Predict
{
    internal class Program
    {
        static void Main(string[] args)
        {

            // 設置GPU (如果有的話)
            //tf.device("/device:GPU:0");
            Environment.SetEnvironmentVariable("TF_GPU_ALLOCATOR", "cuda_malloc_async");

            // 加載MNIST數據集
            var mnist = keras.datasets.mnist;
            (NDArray x_train, NDArray y_train, NDArray x_test, NDArray y_test) = mnist.load_data();
            Console.WriteLine($"Keras數據目錄: {Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)}\\.keras\\datasets");
            // 數據預處理
            x_train = x_train.reshape(new Shape(-1, 28, 28, 1)).astype(np.float32) / 255.0f;
            x_test = x_test.reshape(new Shape(-1, 28, 28, 1)).astype(np.float32) / 255.0f;
            y_train = np_utils.to_categorical(y_train, 10);
            y_test = np_utils.to_categorical(y_test, 10);

            // 構建LeNet-5模型
            var model = keras.Sequential();

            // 添加輸入層
            model.add(keras.layers.InputLayer(input_shape: new Shape(28, 28, 1)));

            // 第一個卷積層
            model.add(keras.layers.Conv2D(
            filters: 6,
            kernel_size: new Shape(5, 5),
            padding: "same",
            activation: "relu"));

            model.add(keras.layers.MaxPooling2D(pool_size: new Shape(2, 2)));

            // 第二個卷積層
            model.add(keras.layers.Conv2D(16, kernel_size: new Shape(5, 5),
                                         activation: "relu"));
            model.add(keras.layers.MaxPooling2D(pool_size: new Shape(2, 2)));

            // 全連接層
            model.add(keras.layers.Flatten());
            model.add(keras.layers.Dense(120, activation: "relu"));
            model.add(keras.layers.Dense(84, activation: "relu"));
            model.add(keras.layers.Dense(10, activation: "softmax"));

            // 編譯模型
            model.compile(optimizer: keras.optimizers.Adam(),
                         loss: keras.losses.CategoricalCrossentropy(),
                         metrics: new[] { "accuracy" });

            // 顯示模型結構
            model.summary();

            // 訓練模型
            model.fit(x_train, y_train,
                     batch_size: 32,
                     epochs: 5,
                     validation_split: 0.1f);

            // 評估模型
            Dictionary<string, float> evaluationResults = model.evaluate(x_test, y_test, return_dict: true);

            if (evaluationResults.TryGetValue("loss", out float loss) &&
                evaluationResults.TryGetValue("accuracy", out float accuracy))
            {
                Console.WriteLine($"Test loss: {loss}, Test accuracy: {accuracy}");
            }
            else
            {
                Console.WriteLine("Could not retrieve loss and accuracy from evaluation results.");
            }

            // 保存模型
            string savedModelPath = @"C:\Users\Nekzat\Downloads\model";
            model.save(savedModelPath, save_format: "h5");
            Console.WriteLine($"Model saved to: {savedModelPath}");

            while (true)
            {

                string? imagePath = Console.ReadLine();
                if (imagePath == null)
                {
                    break;
                }
                try
                {
                    var processedImage = PreprocessImage(imagePath);
                    var predictions = model.predict(processedImage);

                    Console.WriteLine("==============================================================");
                    Console.WriteLine(predictions);
                    Console.WriteLine("==============================================================");
                    // 使用三種不同方法顯示結果
                    PredictMethod1(predictions);
                    Console.WriteLine("==============================================================");

                    PredictMethod2(predictions);
                    Console.WriteLine("==============================================================");

                    PredictMethod3(predictions);


                }
                catch (Exception ex)
                {
                    Console.WriteLine($"預測過程出錯: {ex.Message}");
                }
            }









        }


        // 將方法改為靜態
        private static NDArray PreprocessImage(string imagePath)
        {
            // 讀取圖片
            using (var image = Image.Load<Rgba32>(imagePath))
            {
                // 調整大小為 28x28
                image.Mutate(x => x.Resize(28, 28));

                // 創建數組存儲灰度值
                var gray = new float[28, 28];

                // 轉換為灰度值並正規化
                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        var pixel = image[x, y];
                        // 計算灰度值並正規化到 0-1
                        gray[y, x] = (pixel.R + pixel.G + pixel.B) / (3.0f * 255.0f);
                    }
                }

                // 轉換為 NDArray 並調整形狀
                var imageArray = np.array(gray);
                return imageArray.reshape(new Shape(1, 28, 28, 1));
            }
        }



        private static void PredictMethod1(Tensor predictions)
        {
            var predictedIndex = tf.arg_max(predictions, 1);
            var result = predictedIndex.numpy().ToArray<long>()[0];
            Console.WriteLine($"方法1 - 預測結果: {result}");
        }

        private static void PredictMethod2(Tensor predictions)
        {
            var probabilities = predictions.numpy();
            var maxIndex = np.argmax(probabilities[0]);
            var confidence = probabilities[0][maxIndex.astype(np.int32)];
            Console.WriteLine($"方法2 - 預測結果: {maxIndex}, 置信度: {confidence:P2}");
        }

        private static void PredictMethod3(Tensor predictions)
        {
            var probs = predictions.numpy()[0];
            var topN = np.argsort(-probs)  // 降序排序
                .ToArray<int>()
                .Take(3);  // 取前3個

            Console.WriteLine("方法3 - 前三個可能的結果：");
            foreach (var (index, prob) in topN.Select((v, i) => (v, probs[v])))
            {
                Console.WriteLine($"數字 {index}: {prob:P2}的可能性");
            }
        }



    }



}




/*internal class Program
    {
        static void Main(string[] args)
        {

            // 設置GPU (如果有的話)
            //tf.device("/device:GPU:0");
            Environment.SetEnvironmentVariable("TF_GPU_ALLOCATOR", "cuda_malloc_async");

            // 加載MNIST數據集
            var mnist = keras.datasets.mnist;
            (NDArray x_train, NDArray y_train, NDArray x_test, NDArray y_test) = mnist.load_data();
            Console.WriteLine($"Keras數據目錄: {Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)}\\.keras\\datasets");
            // 數據預處理
            x_train = x_train.reshape(new Shape(-1, 28, 28, 1)).astype(np.float32) / 255.0f;
            x_test = x_test.reshape(new Shape(-1, 28, 28, 1)).astype(np.float32) / 255.0f;
            y_train = np_utils.to_categorical(y_train, 10);
            y_test = np_utils.to_categorical(y_test, 10);

            // 構建LeNet-5模型
            var model = keras.Sequential();

            // 添加輸入層
            model.add(keras.layers.InputLayer(input_shape: new Shape(28, 28, 1)));

            // 第一個卷積層
            model.add(keras.layers.Conv2D(
            filters: 6,
            kernel_size: new Shape(5, 5),
            padding: "same",
            activation: "relu"));
                
            model.add(keras.layers.MaxPooling2D(pool_size: new Shape(2, 2)));

            // 第二個卷積層
            model.add(keras.layers.Conv2D(16, kernel_size: new Shape(5, 5),
                                         activation: "relu"));
            model.add(keras.layers.MaxPooling2D(pool_size: new Shape(2, 2)));

            // 全連接層
            model.add(keras.layers.Flatten());
            model.add(keras.layers.Dense(120, activation: "relu"));
            model.add(keras.layers.Dense(84, activation: "relu"));
            model.add(keras.layers.Dense(10, activation: "softmax"));

            // 編譯模型
            model.compile(optimizer: keras.optimizers.Adam(),
                         loss: keras.losses.CategoricalCrossentropy(),
                         metrics: new[] { "accuracy" });

            // 顯示模型結構
            model.summary();

            // 訓練模型
            model.fit(x_train, y_train,
                     batch_size: 32,
                     epochs: 5,
                     validation_split: 0.1f);

            // 評估模型
            Dictionary<string, float> evaluationResults = model.evaluate(x_test, y_test, return_dict: true);

            if (evaluationResults.TryGetValue("loss", out float loss) &&
                evaluationResults.TryGetValue("accuracy", out float accuracy))
            {
                Console.WriteLine($"Test loss: {loss}, Test accuracy: {accuracy}");
            }
            else
            {
                Console.WriteLine("Could not retrieve loss and accuracy from evaluation results.");
            }

            // 保存模型
            string savedModelPath = @"C:\Users\Nekzat\Downloads\model";
            model.save(savedModelPath, save_format: "h5");
            Console.WriteLine($"Model saved to: {savedModelPath}");

            while (true)
            {
                
                string? imagePath = Console.ReadLine().ToString();
                if (imagePath == null)
                {
                    break;
                }
                try
                {
                    var processedImage = PreprocessImage(imagePath);
                    var predictions = model.predict(processedImage);

                    Console.WriteLine("==============================================================");
                    Console.WriteLine(predictions);
                    Console.WriteLine("==============================================================");
                    // 使用三種不同方法顯示結果
                    PredictMethod1(predictions);
                    Console.WriteLine("==============================================================");

                    PredictMethod2(predictions);
                    Console.WriteLine("==============================================================");

                    PredictMethod3(predictions);


                }
                catch (Exception ex)
                {
                    Console.WriteLine($"預測過程出錯: {ex.Message}");
                }
            }









        }


        // 將方法改為靜態
        private static NDArray PreprocessImage(string imagePath)
        {
            // 讀取圖片
            using (var image = Image.Load<Rgba32>(imagePath))
            {
                // 調整大小為 28x28
                image.Mutate(x => x.Resize(28, 28));

                // 創建數組存儲灰度值
                var gray = new float[28, 28];

                // 轉換為灰度值並正規化
                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        var pixel = image[x, y];
                        // 計算灰度值並正規化到 0-1
                        gray[y, x] = (pixel.R + pixel.G + pixel.B) / (3.0f * 255.0f);
                    }
                }

                // 轉換為 NDArray 並調整形狀
                var imageArray = np.array(gray);
                return imageArray.reshape(new Shape(1, 28, 28, 1));
            }
        }



        private static void PredictMethod1(Tensor predictions)
        {
            var predictedIndex = tf.arg_max(predictions, 1);
            var result = predictedIndex.numpy().ToArray<long>()[0];
            Console.WriteLine($"方法1 - 預測結果: {result}");
        }

        private static void PredictMethod2(Tensor predictions)
        {
            var probabilities = predictions.numpy();
            var maxIndex = np.argmax(probabilities[0]);
            var confidence = probabilities[0][maxIndex.astype(np.int32)];
            Console.WriteLine($"方法2 - 預測結果: {maxIndex}, 置信度: {confidence:P2}");
        }

        private static void PredictMethod3(Tensor predictions)
        {
            var probs = predictions.numpy()[0];
            var topN = np.argsort(-probs)  // 降序排序
                .ToArray<int>()
                .Take(3);  // 取前3個

            Console.WriteLine("方法3 - 前三個可能的結果：");
            foreach (var (index, prob) in topN.Select((v, i) => (v, probs[v])))
            {
                Console.WriteLine($"數字 {index}: {prob:P2}的可能性");
            }
        }



    }*/
/*
        
*/

















/*
        internal class Program
        {
            static void Main(string[] args)
            {

                tf.device("/device:GPU:0");
                Environment.SetEnvironmentVariable("TF_GPU_ALLOCATOR", "cuda_malloc_async");
                tf.set_random_seed(1);
                // 加载MNIST数据集
                var mnist = keras.datasets.mnist;
                (NDArray x_train, NDArray y_train, NDArray x_test, NDArray y_test) = mnist.load_data();
                // 数据预处理
                x_train = x_train.reshape(new Shape(-1, 28, 28, 1)).astype(np.float32) / 255.0f;
                x_test = x_test.reshape(new Shape(-1, 28, 28, 1)).astype(np.float32) / 255.0f;
                y_train = np_utils.to_categorical(y_train, 10);
                y_test = np_utils.to_categorical(y_test, 10);
                // 构建VGG16模型
                var model = keras.Sequential();
                // 第一个卷积块
                model.add(keras.layers.InputLayer(input_shape: new Shape(28, 28, 1)));
                model.add(keras.layers.Conv2D(64, kernel_size: new Shape(3, 3), padding: "same", activation: "relu"));
                model.add(keras.layers.Conv2D(64, kernel_size: new Shape(3, 3), padding: "same", activation: "relu"));
                model.add(keras.layers.MaxPooling2D(pool_size: new Shape(2, 2), strides: new Shape(2, 2)));
                // 第二个卷积块
                model.add(keras.layers.Conv2D(128, kernel_size: new Shape(3, 3), padding: "same", activation: "relu"));
                model.add(keras.layers.Conv2D(128, kernel_size: new Shape(3, 3), padding: "same", activation: "relu"));
                model.add(keras.layers.MaxPooling2D(pool_size: new Shape(2, 2), strides: new Shape(2, 2)));
                // 第三个卷积块

                model.add(keras.layers.Conv2D(256, kernel_size: new Shape(3, 3), padding: "same", activation: "relu"));
                model.add(keras.layers.Conv2D(256, kernel_size: new Shape(3, 3), padding: "same", activation: "relu"));
                model.add(keras.layers.Conv2D(256, kernel_size: new Shape(3, 3), padding: "same", activation: "relu"));
                model.add(keras.layers.MaxPooling2D(pool_size: new Shape(2, 2), strides: new Shape(2, 2)));
                // 全连接层
                model.add(keras.layers.Flatten());
                model.add(keras.layers.Dense(4096, activation: "relu"));
                //model.add(keras.layers.Dense(4096, activation: "relu"));
                model.add(keras.layers.Dense(10, activation: "softmax"));
                // 编译模型
                model.compile(optimizer: keras.optimizers.Adam(),
                              loss: keras.losses.CategoricalCrossentropy(),
                              metrics: new[] { "accuracy" });
                model.summary();
                // 训练模型
                model.fit(x_train, y_train, batch_size: 16, epochs: 2, validation_split: 0.1f);
                // 评估模型
                Dictionary<string, float> evaluationResults = model.evaluate(x_test, y_test, return_dict: true);
                if (evaluationResults.TryGetValue("loss", out float loss) &&
                    evaluationResults.TryGetValue("accuracy", out float accuracy))
                {
                    Console.WriteLine($"Test loss: {loss}, Test accuracy: {accuracy}  ");
                }
                else
                {
                    Console.WriteLine("Could not retrieve loss and accuracy from evaluation results.");
                }
                string savedModelPath = @"C:\Users\Nekzat\Downloads\model\mnist_model";
                model.save(savedModelPath, save_format: "tf");
                Console.WriteLine($"Model saved to: {savedModelPath}");



            }
        }
    */

/*
 static void Main(string[] args)
{
try
{
    string modelPath = @"path_to_your_pb_file.pb";
    AnalyzeModel(modelPath);
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
Console.ReadLine();
}

static void AnalyzeModel(string modelPath)
{
var graph = new Graph();
graph.Import(File.ReadAllBytes(modelPath));

Console.WriteLine("Model Structure Analysis:");
Console.WriteLine("------------------------");

var ops = graph.get_operations().ToList();
Console.WriteLine($"Total operations: {ops.Count}");

foreach (var op in ops)
{
    try
    {
        Console.WriteLine($"\nOperation: {op}");

        var outputs = op.outputs;
        if (outputs != null && outputs.Length > 0)
        {
            Console.WriteLine($"Number of outputs: {outputs.Length}");

            foreach (var output in outputs)
            {
                var shape = output.shape;
                if (shape != null)
                {
                    Console.WriteLine($"Shape: {string.Join(",", shape.dims)}");
                }
            }
        }

        // 嘗試取得所有可用的屬性
        foreach (var prop in op.GetType().GetProperties())
        {
            try
            {
                var value = prop.GetValue(op);
                Console.WriteLine($"{prop.Name}: {value}");
            }
            catch
            {
                Console.WriteLine($"{prop.Name}: Unable to get value");
            }
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error analyzing operation: {ex.Message}");
    }
    Console.WriteLine("------------------------");
}
}
 */