using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SentimentAnalysis
{
    public class SentimentData
    {
        // CSV layout: sentiment,title,review_text
        [LoadColumn(2)]
        public string? SentimentText;

        // After preprocessing, Label column will contain boolean values (true/false)
        [LoadColumn(0), ColumnName("Label")]
        public bool Sentiment;
    }

    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
