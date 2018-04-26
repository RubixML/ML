<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Ridge;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use Rubix\Engine\Metrics\Reports\RegressionAnalysis;
use League\Csv\Reader;

$alpha = $argv[1] ?? 0.5;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ MPG Estimator using Ridge Regression                ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/auto-mpg.csv')->setDelimiter(',')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.20);

$estimator = new Prototype(new Pipeline(new Ridge($alpha), [
    new MissingDataImputer('?'), new ZScaleStandardizer(),
]), [
    new RegressionAnalysis(),
]);

$estimator->train($training);

$estimator->test($testing);
