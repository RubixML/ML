<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\NaiveBayes;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Transformers\TextNormalizer;
use Rubix\Engine\Transformers\StopWordFilter;
use Rubix\Engine\Transformers\TfIdfTransformer;
use Rubix\Engine\Metrics\Reports\ConfusionMatrix;
use Rubix\Engine\Transformers\TokenCountVectorizer;
use Rubix\Engine\Transformers\BlanketCharacterFilter;
use Rubix\Engine\Metrics\Reports\ClassificationReport;
use Rubix\Engine\Transformers\Tokenizers\WordTokenizer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Sentiment Analyzer using Naive Bayes                ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/sentiment.csv')->setDelimiter(',')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.2);

$estimator = new Prototype(new Pipeline(new NaiveBayes(), [
    new BlanketCharacterFilter(BlanketCharacterFilter::SPECIAL),
    new TextNormalizer(),
    new StopWordFilter(file(dirname(__DIR__) . '/datasets/stopwords.txt')),
    new TokenCountVectorizer(1000, new WordTokenizer()),
    new TfIdfTransformer(),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

$estimator->test($testing);
