<?php

use Rubix\Engine\WeightedDataset;
use PHPUnit\Framework\TestCase;

class WeightedDatasetTest extends TestCase
{
    protected $data;

    public function setUp()
    {
        $samples = [
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
        ];

        $outcomes = ['not a monster', 'monster', 'not a monster', 'monster'];

        $weights = [0, 0, 1, 0];

        $this->dataset = new WeightedDataset($samples, $outcomes, $weights);
    }

    public function test_build_weighted_dataset()
    {
        $this->assertInstanceOf(WeightedDataset::class, $this->dataset);
    }

    public function test_generate_random_weighted_subset_with_replacement()
    {
        $samples = $this->dataset->generateRandomWeightedSubsetWithReplacement(0.25)->all();

        $this->assertEquals(['nice', 'rough', 'friendly', 'not a monster'], $samples[0]);
    }
}
