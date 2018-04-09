<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\NaiveBayes;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Preprocessors\TextNormalizer;
use Rubix\Engine\Preprocessors\TokenCountVectorizer;
use Rubix\Engine\Preprocessors\BlanketCharacterFilter;
use Rubix\Engine\Preprocessors\Tokenizers\WordTokenizer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Sentiment Analyzer using Naive Bayes                ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/sentiment.csv')->setDelimiter(',')->getRecords();

$stopWords = file(dirname(__DIR__) . '/datasets/stopwords.txt');

$dataset = SupervisedDataset::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.3);

$prototype = new Prototype(new Pipeline(new NaiveBayes(), [
    new TextNormalizer(),
    new BlanketCharacterFilter(),
    new TokenCountVectorizer(new WordTokenizer(), $stopWords),
]), [new Accuracy()]);

$prototype->train($training);

$prototype->test($testing);
