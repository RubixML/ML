<?php

namespace Rubix\Tests\Regressors;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\Regressor;
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Transformers\Strategies\BlurryMean;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DummyRegressorTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $samples = [
            [4, 119, 82, 2720], [4, 120, 79, 2625], [4, 135, 84, 2295],
            [4, 97, 52, 2130], [4, 140, 86, 2790], [4, 151, 90, 2950],
            [4, 135, 84, 2370], [4, 144, 96, 2665], [6, 232, 112, 2835],
            [4, 156, 92, 2585], [6, 262, 85, 3015], [6, 181, 110, 2945],
            [4, 91, 67, 1995], [4, 108, 70, 2125], [4, 105, 63, 2125],
            [4, 91, 68, 1970], [4, 105, 74, 1980], [4, 140, 92, 2865],
            [4, 151, 90, 2735], [4, 135, 84, 2525], [4, 112, 88, 2395],
            [4, 112, 88, 2640], [6, 225, 85, 3465], [6, 200, 88, 3060],
            [8, 350, 105, 3725], [6, 231, 110, 3415], [6, 146, 120, 2930],
            [6, 168, 116, 2900], [6, 145, 76, 3160], [4, 141, 80, 3230],
            [4, 120, 74, 2635], [4, 119, 100, 2615], [4, 108, 75, 2350],
            [4, 107, 75, 2210], [4, 105, 74, 2190], [4, 98, 65, 2380],
            [4, 98, 65, 2045], [4, 105, 63, 2215], [4, 91, 68, 1985],
            [4, 89, 62, 2050], [4, 85, 65, 1975], [4, 97, 67, 2065],
            [4, 81, 60, 1760], [6, 173, 110, 2725], [4, 122, 88, 2500],
            [6, 168, 132, 2910], [4, 97, 67, 2145], [4, 91, 67, 1850],
            [4, 146, 67, 3250], [5, 121, 67, 2950], [4, 90, 48, 2335],
            [4, 90, 48, 2085], [4, 85, 65, 2110], [4, 156, 105, 2800],
            [4, 86, 65, 2110], [6, 225, 90, 3003], [4, 151, 90, 3003],
            [4, 86, 65, 2019], [4, 98, 70, 2120], [4, 89, 60, 1968],
            [4, 98, 76, 2144], [4, 151, 90, 2556], [6, 173, 115, 2700],
            [6, 173, 115, 2595], [4, 151, 90, 2670], [4, 91, 69, 2130],
        ];

        $labels = [
            31, 28, 32, 44, 27, 27, 36, 32, 22, 26, 38, 25, 38, 36, 38, 31, 36,
            24, 27, 29, 34, 27, 18, 20, 27, 22, 24, 25, 31, 28, 32, 33, 32, 34,
            33, 30, 34, 35, 34, 38, 37, 32, 35, 24, 35, 33, 34, 45, 30, 36, 43,
            44, 41, 28, 47, 19, 24, 37, 32, 38, 41, 34, 27, 29, 28, 37,
        ];

        $this->training = new Labeled($samples, $labels);

        $this->testing = new Labeled([
            [4, 156, 92, 2620], [4, 107, 72, 2290],
        ], [
            26, 32,
        ]);

        $this->estimator = new DummyRegressor(new BlurryMean());
    }

    public function test_build_dummy_regressor()
    {
        $this->assertInstanceOf(DummyRegressor::class, $this->estimator);
        $this->assertInstanceOf(Regressor::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0], '', INF);
        $this->assertEquals($this->testing->label(1), $predictions[1], '', INF);
    }

    public function test_train_with_unlabeled()
    {
        $dataset = new Unlabeled([['bad']]);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train($dataset);
    }
}
