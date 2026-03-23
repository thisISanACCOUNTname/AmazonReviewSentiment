using Microsoft.ML;
using Microsoft.ML.Data;
using SentimentAnalysis;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;
// Small notes:
// - This file is a simple console program that trains and evaluates
//   a binary sentiment classifier using ML.NET.
// - Top-level statements are used so there is no explicit Program class.
// Path to the original CSV with raw reviews.
string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train-reviews-micro - train-reviews-micro.csv");

// Create the ML.NET context (shared across operations).
MLContext mlContext = new MLContext();
// Load and preprocess data, returning training and test IDataView objects.
var splitDataView = LoadData(mlContext);
// LoadData: read raw CSV, convert labels to boolean text, write a processed CSV,
// then load it into an IDataView and split it into train and test sets.
(IDataView TrainSet, IDataView TestSet) LoadData(MLContext mlContext)
{
    // Preprocess original CSV and write a temporary CSV where the sentiment column
    // is converted to boolean strings (true for positive=2, false for negative=1).
    var original = _dataPath;
    var processed = Path.Combine(Environment.CurrentDirectory, "Data", "train-reviews-processed.csv");
    if (!File.Exists(processed))
    {
        var lines = File.ReadAllLines(original);
        if (lines.Length > 0)
        {
            var header = lines[0];
            var output = new List<string>(lines.Length);
            output.Add(header); // keep header
            for (int i = 1; i < lines.Length; i++)
            {
                var line = lines[i];
                // Split only on first two commas to preserve commas in text
                var parts = line.Split(new[] { ',' }, 3);
                if (parts.Length >= 3)
                {
                    var label = parts[0].Trim();
                    var mapped = (label == "2") ? "true" : "false";
                    output.Add(string.Join(",", mapped, parts[1], parts[2]));
                }
                else
                {
                    output.Add(line);
                }
            }
            File.WriteAllLines(processed, output);
        }
    }

    // Load processed CSV with boolean Label column into a typed IDataView
    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(processed, hasHeader: true, separatorChar: ',');

    // Use built-in TrainTestSplit
    var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    return (split.TrainSet, split.TestSet);
}
// BuildAndTrainModel: define the data processing + training pipeline and fit it
ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
    
    // Train the model on the training set
    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();

    return model;
}
// Run evaluation and example predictions
Evaluate(mlContext, model, splitDataView.TestSet);
UseModelWithSingleItem(mlContext, model);
UseModelWithBatchItems(mlContext, model);


// Evaluate: compute metrics on the test set and print class-level correctness
void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    // Inspect actual label distribution in test set. The label column might be
    // stored as float (e.g., 2 for positive) or as bool, so we handle both.
    var schema = splitTestSet.Schema;
    int labelIndex = -1;
    for (int i = 0; i < schema.Count; i++)
        if (schema[i].Name == "Label" || schema[i].Name == "Sentiment") { labelIndex = i; break; }
    if (labelIndex >= 0)
    {
        var labelCol = schema[labelIndex];
        int pos = 0, neg = 0;
        using (var cursor = splitTestSet.GetRowCursor(new[] { labelCol }))
        {
            if (labelCol.Type.RawType == typeof(float))
            {
                var get = cursor.GetGetter<float>(labelCol);
                float v = 0f;
                while (cursor.MoveNext()) { get(ref v); if (Math.Abs(v - 2f) < 0.001f) pos++; else neg++; }
            }
            else if (labelCol.Type.RawType == typeof(bool))
            {
                var get = cursor.GetGetter<bool>(labelCol);
                bool v = false;
                while (cursor.MoveNext()) { get(ref v); if (v) pos++; else neg++; }
            }
        }
        Console.WriteLine($"Test set distribution -> Positive: {pos}, Negative: {neg}");
    }
    // Run the model on the test set to produce predictions
    IDataView predictions = model.Transform(splitTestSet);

    // Evaluate the model and show metrics
    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");

    // Additionally compute percentage of correctly labeled positive and negative reviews
    int totalPos = 0, totalNeg = 0, correctPos = 0, correctNeg = 0;
    var schemaPred = predictions.Schema;
    // Find column indices by name to avoid relying on API surface that may not be
    // available in all ML.NET package versions.
    int labelColIndex = -1, predColIndex = -1;
    for (int i = 0; i < schemaPred.Count; i++)
    {
        if (schemaPred[i].Name == "Label") labelColIndex = i;
        if (schemaPred[i].Name == "PredictedLabel") predColIndex = i;
    }

    // If both label and predicted label columns exist, iterate rows to count
    // how many positives/negatives were predicted correctly.
    if (labelColIndex >= 0 && predColIndex >= 0)
    {
        var labelCol = schemaPred[labelColIndex];
        var predCol = schemaPred[predColIndex];

        using (var cursor = predictions.GetRowCursor(new[] { labelCol, predCol }))
        {
            // Handle boolean label column
            if (labelCol.Type.RawType == typeof(bool))
            {
                var getLabel = cursor.GetGetter<bool>(labelCol);
                var getPred = cursor.GetGetter<bool>(predCol);
                bool lab = false, pred = false;
                while (cursor.MoveNext())
                {
                    getLabel(ref lab);
                    getPred(ref pred);
                    if (lab)
                    {
                        totalPos++;
                        if (pred) correctPos++;
                    }
                    else
                    {
                        totalNeg++;
                        if (!pred) correctNeg++;
                    }
                }
            }
            else if (labelCol.Type.RawType == typeof(float))
            {
                // Some pipelines use float labels (e.g., 2 for positive); treat ~2.0 as positive
                var getLabel = cursor.GetGetter<float>(labelCol);
                var getPred = cursor.GetGetter<bool>(predCol);
                float labf = 0f; bool pred = false;
                while (cursor.MoveNext())
                {
                    getLabel(ref labf);
                    getPred(ref pred);
                    bool labBool = Math.Abs(labf - 2f) < 0.001f;
                    if (labBool)
                    {
                        totalPos++;
                        if (pred) correctPos++;
                    }
                    else
                    {
                        totalNeg++;
                        if (!pred) correctNeg++;
                    }
                }
            }
        }
    }

    // Print class-level correctness summaries
    Console.WriteLine();
    Console.WriteLine("Classification correctness by class");
    Console.WriteLine("-----------------------------------");
    if (totalPos > 0)
        Console.WriteLine($"Positive correctly labeled: {(double)correctPos / totalPos:P2} ({correctPos}/{totalPos})");
    else
        Console.WriteLine("Positive correctly labeled: N/A (no positive examples in test set)");

    if (totalNeg > 0)
        Console.WriteLine($"Negative correctly labeled: {(double)correctNeg / totalNeg:P2} ({correctNeg}/{totalNeg})");
    else
        Console.WriteLine("Negative correctly labeled: N/A (no negative examples in test set)");

    Console.WriteLine("=============== End of model evaluation ===============");
}

// UseModelWithSingleItem: create one temporary CSV row, run the model and
// print the single prediction. This avoids using dynamic codegen or
// PredictionEngine for the example.
void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    // Create a small temporary CSV to load the single sample without using PredictionEngine
    var processed = Path.Combine(Environment.CurrentDirectory, "Data", "train-reviews-processed.csv");
    var tempPath = Path.Combine(Environment.CurrentDirectory, "Data", "predict-single.csv");
    string header = "sentiment,title,review_text";
    if (File.Exists(processed))
        header = File.ReadLines(processed).First();

    // write a single-line CSV (label is dummy false)
    File.WriteAllLines(tempPath, new[] { header, $"false,,This was a very bad steak" });

    IDataView singleData = mlContext.Data.LoadFromTextFile<SentimentData>(tempPath, hasHeader: true, separatorChar: ',');
    IDataView predictions = model.Transform(singleData);

    // Read predicted fields via cursor to avoid CreatePredictionEngine/CreateEnumerable
    var schema = predictions.Schema;
    var predCol = schema["PredictedLabel"];
    var probCol = schema["Probability"];
    var textCol = schema["SentimentText"];

    using (var cursor = predictions.GetRowCursor(new[] { predCol, probCol, textCol }))
    {
        var getPred = cursor.GetGetter<bool>(predCol);
        var getProb = cursor.GetGetter<float>(probCol);
        var getText = cursor.GetGetter<ReadOnlyMemory<char>>(textCol);
        bool p = false; float prob = 0f; ReadOnlyMemory<char> txt = default;
        while (cursor.MoveNext())
        {
            getText(ref txt);
            getPred(ref p);
            getProb(ref prob);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {txt.ToString()} | Prediction: {(p ? "Positive" : "Negative")} | Probability: {prob} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
    }
}

// UseModelWithBatchItems: write a small CSV with multiple examples, run the
// model and print predictions for each row.
void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    // Build a temporary CSV for batch predictions and load via LoadFromTextFile to avoid dynamic code gen
    var processed = Path.Combine(Environment.CurrentDirectory, "Data", "train-reviews-processed.csv");
    var tempPath = Path.Combine(Environment.CurrentDirectory, "Data", "predict-batch.csv");
    string header = "sentiment,title,review_text";
    if (File.Exists(processed))
        header = File.ReadLines(processed).First();

    var lines = new List<string> { header };
    lines.Add("false,,This was a horrible meal");
    lines.Add("false,,I love this spaghetti.");
    File.WriteAllLines(tempPath, lines);

    IDataView batchData = mlContext.Data.LoadFromTextFile<SentimentData>(tempPath, hasHeader: true, separatorChar: ',');
    IDataView predictions = model.Transform(batchData);

    var schema = predictions.Schema;
    var predCol = schema["PredictedLabel"];
    var probCol = schema["Probability"];
    var textCol = schema["SentimentText"];

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

    using (var cursor = predictions.GetRowCursor(new[] { predCol, probCol, textCol }))
    {
        var getPred = cursor.GetGetter<bool>(predCol);
        var getProb = cursor.GetGetter<float>(probCol);
        var getText = cursor.GetGetter<ReadOnlyMemory<char>>(textCol);
        bool p = false; float prob = 0f; ReadOnlyMemory<char> txt = default;
        while (cursor.MoveNext())
        {
            getText(ref txt);
            getPred(ref p);
            getProb(ref prob);
            Console.WriteLine($"Sentiment: {txt.ToString()} | Prediction: {(p ? "Positive" : "Negative")} | Probability: {prob} ");
        }
    }
    Console.WriteLine("=============== End of predictions ===============");
}
