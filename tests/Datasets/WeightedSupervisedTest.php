<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Datasets\WeightedSupervised;
use PHPUnit\Framework\TestCase;

class WeightedSupervisedTest extends TestCase
{
    protected $dataset;

    public function setUp()
    {
        $samples = [
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
        ];

        $outcomes = ['not monster', 'monster', 'not monster', 'monster'];

        $weights = [0, 0, 1, 0];

        $this->dataset = new WeightedSupervised($samples, $outcomes, $weights);
    }

    public function test_build_weighted_supervised_dataset()
    {
        $this->assertInstanceOf(WeightedSupervised::class, $this->dataset);
        $this->assertInstanceOf(Supervised::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
    }

    public function test_generate_random_subset_with_replacement()
    {
        $samples = $this->dataset->generateRandomSubsetWithReplacement(0.25)->all();

        $this->assertEquals(['nice', 'rough', 'friendly', 'not monster'], $samples[0]);
    }
}
