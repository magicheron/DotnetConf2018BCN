using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Media;
using Windows.Media.Core;
using Windows.Media.Playback;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.UI.Input.Inking;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;

// La plantilla de elemento Página en blanco está documentada en https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0xc0a

namespace DotnetConf.ML.UWP
{
    /// <summary>
    /// Página vacía que se puede usar de forma independiente o a la que se puede navegar dentro de un objeto Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private MediaPlayer mediaPlayer;
        private mnistModel mnistModel = new mnistModel();
        private mnistInput mnistInput = new mnistInput();
        private mnistOutput mnistOutput = new mnistOutput();
        //private LearningModelSession    session;
        private Helper helper = new Helper();
        RenderTargetBitmap renderBitmap = new RenderTargetBitmap();

        public MainPage()
        {
            this.InitializeComponent();
            this.InitializeDrawing();
            this.LoadModel();
        }

        private void InitializeDrawing()
        {
            // Set supported inking device types.
            this.inkCanvas.InkPresenter.InputDeviceTypes = Windows.UI.Core.CoreInputDeviceTypes.Mouse | Windows.UI.Core.CoreInputDeviceTypes.Pen | Windows.UI.Core.CoreInputDeviceTypes.Touch;
            this.inkCanvas.InkPresenter.UpdateDefaultDrawingAttributes(
                new Windows.UI.Input.Inking.InkDrawingAttributes()
                {
                    Color = Windows.UI.Colors.White,
                    Size = new Size(22, 22),
                    IgnorePressure = true,
                    IgnoreTilt = true,
                }
            );

            this.inkCanvas.InkPresenter.StrokesCollected += StrokesCollected;

            // Set supported inking device types.
            this.inkCanvas2.InkPresenter.InputDeviceTypes = Windows.UI.Core.CoreInputDeviceTypes.Mouse | Windows.UI.Core.CoreInputDeviceTypes.Pen | Windows.UI.Core.CoreInputDeviceTypes.Touch;
            this.inkCanvas2.InkPresenter.UpdateDefaultDrawingAttributes(
                new Windows.UI.Input.Inking.InkDrawingAttributes()
                {
                    Color = Windows.UI.Colors.White,
                    Size = new Size(22, 22),
                    IgnorePressure = true,
                    IgnoreTilt = true,
                }
            );

            this.inkCanvas2.InkPresenter.StrokesCollected += StrokesCollected;
        }

        private async void LoadModel()
        {
            //Load a machine learning model
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/mnist.onnx"));
            mnistModel = await mnistModel.CreateFromStreamAsync(modelFile as IRandomAccessStreamReference);
        }

        private void StrokesCollected(InkPresenter sender, InkStrokesCollectedEventArgs args)
        {
            this.Recognize(sender, null);
        }

        private async void Recognize(object sender, RoutedEventArgs e)
        {
            var firstNumber = await this.Evaluate(this.inkGrid, this.inkCanvas);
            var secondNumber = await this.Evaluate(this.inkGrid2, this.inkCanvas2);

            var result = firstNumber + secondNumber;

            //Display the results
            numberLabel.Text = result.ToString();
        }

        private async Task<int> Evaluate(Grid inkGrid, InkCanvas canvas)
        {
            if (canvas.InkPresenter.StrokeContainer.BoundingRect.Height == 0) return 0;

            //Bind model input with contents from InkCanvas
            VideoFrame vf = await helper.GetHandWrittenImage(inkGrid);
            mnistInput.Input3 = ImageFeatureValue.CreateFromVideoFrame(vf);

            //Evaluate the model
            mnistOutput = await mnistModel.EvaluateAsync(mnistInput);

            //Convert output to datatype
            IReadOnlyList<float> VectorImage = mnistOutput.Plus214_Output_0.GetAsVectorView();
            IList<float> ImageList = VectorImage.ToList();

            //LINQ query to check for highest probability digit
            var maxIndex = ImageList.IndexOf(ImageList.Max());

            return maxIndex;
        }

        private void Clear(object sender, RoutedEventArgs e)
        {
            this.inkCanvas.InkPresenter.StrokeContainer.Clear();
            this.inkCanvas2.InkPresenter.StrokeContainer.Clear();
            numberLabel.Text = "";
        }

    }
}
