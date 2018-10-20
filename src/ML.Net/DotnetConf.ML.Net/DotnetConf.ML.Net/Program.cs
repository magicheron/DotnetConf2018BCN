using Microsoft.ML.Core.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Tools;
using System;
using System.IO;
using Microsoft.ML.Runtime;
using System.Linq;

namespace DotnetConf.ML.Net
{
    class Program
    {
        static void Main(string[] args)
        {
            var modelFile = "model.onnx";
            var definitions = "synset.txt";

            using (var env = new ConsoleEnvironment(null, false, 0, 1, null, null))
            {
                var imageHeight = 224;
                var imageWidth = 224;
                var dataFile = @"images.tsv";
                var imageFolder = Path.GetDirectoryName(@"C:\Projects\DotnetConf2018BCN\kitten.jpg");

                var data = TextLoader.CreateReader(env, ctx => (
                    imagePath: ctx.LoadText(0),
                    name: ctx.LoadText(1)))
                    .Read(new MultiFileSource(dataFile));
                
                // Note that CamelCase column names are there to match the TF graph node names.
                var pipe = data.MakeNewEstimator()
                    .Append(row => (
                        name: row.name,
                        data_0: row.imagePath.LoadAsImage(imageFolder).Resize(imageHeight, imageWidth).ExtractPixels(interleaveArgb: true)))
                    .Append(row => (row.name, softmaxout_1: row.data_0.ApplyOnnxModel(modelFile)));
                
                var result = pipe.Fit(data).Transform(data).AsDynamic;
                result.Schema.TryGetColumnIndex("softmaxout_1", out int output);
                using (var cursor = result.GetRowCursor(col => col == output))
                {
                    var buffer = default(VBuffer<float>);
                    var getter = cursor.GetGetter<VBuffer<float>>(output);
                    while (cursor.MoveNext())
                    {
                        getter(ref buffer);
                    }

                    var maxProbability = buffer.Values.Max();

                    var predictionIndex = buffer.Values.ToList().IndexOf(maxProbability);

                    var definitionLines = File.ReadAllLines(definitions);

                    Console.WriteLine($"{definitionLines[predictionIndex]}; --> Probability: {maxProbability*100}%");
                    Console.ReadLine();
                }
            }

        }
    }
}
