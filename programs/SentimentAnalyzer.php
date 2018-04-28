<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\MultiLayerPerceptron;
use Rubix\Engine\Persisters\Filesystem;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\NeuralNet\LearningRates\Adam;
use Rubix\Engine\Transformers\TextNormalizer;
use Rubix\Engine\Transformers\StopWordFilter;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\Transformers\TfIdfTransformer;
use Rubix\Engine\Transformers\TokenCountVectorizer;
use Rubix\Engine\Transformers\BlanketCharacterFilter;
use Rubix\Engine\Transformers\Tokenizers\WordTokenizer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Sentiment Analyzer using Multi Layer Perceptron     ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/sentiment.csv')->setDelimiter(',')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.2);

$estimator = new Prototype(new Pipeline(new MultiLayerPerceptron(1000, [10, 10, 10], $dataset->labels(), 30, 5, new Adam(0.01)), [
    new BlanketCharacterFilter(BlanketCharacterFilter::SPECIAL),
    new TextNormalizer(),
    new TokenCountVectorizer(1000, new WordTokenizer()),
    new TfIdfTransformer(),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

$estimator->test($testing);

$persister = new Filesystem(dirname(__DIR__) . '/models/sentiment.model');

$persister->save($estimator);
