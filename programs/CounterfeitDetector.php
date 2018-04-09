<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\CART;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Preprocessors\L1Normalizer;
use League\Csv\Reader;

$minSamples = $argv[1] ?? 3;
$maxDepth = $argv[2] ?? 300;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Counterfeit Banknote Detector using CART            ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/banknotes.csv')->setDelimiter(',')->getRecords();

$dataset = SupervisedDataset::fromIterator($dataset);

list ($training, $testing) = $dataset->randomize()->split(0.3);

$prototype = new Prototype(new Pipeline(new CART($minSamples, $maxDepth), [new L1Normalizer()]), [new Accuracy()]);

$prototype->train($training);

$prototype->test($testing);
