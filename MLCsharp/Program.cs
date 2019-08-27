using System;
using Tensorflow;
using NumSharp;
using System.Collections.Generic;

namespace MLCsharp
{

    class Program
    {
        public static IEnumerable<(T, T)> zip<T>(NDArray t1, NDArray t2) where T : unmanaged
        {
            var a = t1.Data<T>();
            var b = t2.Data<T>();
            for(int i = 0; i < a.Length; i++) {
                yield return (a[i], b[i]);
            }
        }

        public int training_epochs = 1000;

        // Parameters
        float learning_rate = 0.01f;
        int display_step = 50;
        NDArray train_X, train_Y;
        int n_samples;

        public void prepareData()
        {
            var x = np.linspace(-10.0f, 10.0f, 200, dtype: np.float32);
            var n = np.random.normal(0.0f, 4.0f, 200);
            //noise must be expicitly cast to float data type. Unfortunatly np.random.normal does not support casting a type
            var noise = np.array(n.GetData<float>());
            var yvals = new List<float>();
            for (int i = 0; i < x.size; i++)
            {
                yvals.Add(x[i] + noise[i]);
            }
            var y = np.array<float>(yvals.ToArray());
            this.train_X = x;
            this.train_Y = y;
            this.n_samples = train_X.shape[0];
        }
        public bool Run()
        {
            //a linear regression example to explore the basics of tensorflow
            prepareData();

            // tf Graph Input
            var X = tf.placeholder(tf.float32);
            var Y = tf.placeholder(tf.float32);

            // Set model weights 
            // We can set a fixed init value in order to debug
            // var rnd1 = rng.randn<float>();
            // var rnd2 = rng.randn<float>();
            var W = tf.Variable(-0.06f, name: "weight");
            var b = tf.Variable(-0.73f, name: "bias");

            // Construct a linear model
            var pred = tf.add(tf.multiply(X, W), b);

            // Mean squared error
            var cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * n_samples);

            // Gradient descent
            // Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();
            using (var sess = tf.Session())
            {
                // Run the initializer
                sess.run(init);

                // Fit all training data
                for (int epoch = 0; epoch < training_epochs; epoch++)
                {
                    foreach (var (x, y) in zip<float>(train_X, train_Y))
                    {
                        sess.run(optimizer,
                            new FeedItem(X, x),
                            new FeedItem(Y, y));
                    }

                    // Display logs per epoch step
                    if ((epoch + 1) % display_step == 0)
                    {
                        var c = sess.run(cost,
                            new FeedItem(X, train_X),
                            new FeedItem(Y, train_Y));
                        Console.WriteLine($"Epoch: {epoch + 1} cost={c} " + $"W={sess.run(W)} b={sess.run(b)}");
                    }
                }

                Console.WriteLine("Optimization Finished!");
                var training_cost = sess.run(cost,
                    new FeedItem(X, train_X),
                    new FeedItem(Y, train_Y));
                Console.WriteLine($"Training cost={training_cost} W={sess.run(W)} b={sess.run(b)}");

                // Testing example
                var test_X = np.array(6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f);
                var test_Y = np.array(1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f);
                Console.WriteLine("Testing... (Mean square loss Comparison)");
                var testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * test_X.shape[0]),
                    new FeedItem(X, test_X),
                    new FeedItem(Y, test_Y));
                Console.WriteLine($"Testing cost={testing_cost}");
                var diff = Math.Abs((float)training_cost - (float)testing_cost);
                Console.WriteLine($"Absolute mean square loss difference: {diff}");

                return diff < 0.01;
            }
        }
        static void Main(string[] args)
        {
            var program = new Program();
            program.Run();
        }
    }
}
