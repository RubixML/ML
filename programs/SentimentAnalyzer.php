<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\MCC;
use Rubix\Engine\NaiveBayes;
use Rubix\Engine\Tests\F1Score;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Tests\Informedness;
use Rubix\Engine\Transformers\TextNormalizer;
use Rubix\Engine\Transformers\StopWordFilter;
use Rubix\Engine\Transformers\TokenCountVectorizer;
use Rubix\Engine\Transformers\BlanketCharacterFilter;
use Rubix\Engine\Transformers\Tokenizers\WordTokenizer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Sentiment Analyzer using Naive Bayes                ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/sentiment.csv')->setDelimiter(',')->getRecords();

$dataset = SupervisedDataset::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.2);

$prototype = new Prototype(new Pipeline(new NaiveBayes(), [
    new StopWordFilter(file(dirname(__DIR__) . '/datasets/stopwords.txt')),
    new BlanketCharacterFilter(BlanketCharacterFilter::SPECIAL),
    new TextNormalizer(),
    new TokenCountVectorizer(400, new WordTokenizer()),
]), [
    new Accuracy(),
    new F1Score(),
    new MCC(),
    new Informedness(),
]);

$prototype->train($training);

$prototype->test($testing);
