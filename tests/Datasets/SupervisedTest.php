<?php

use Rubix\Engine\Datasets\Supervised;
use PHPUnit\Framework\TestCase;

class SupervisedTest extends TestCase
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

        $this->dataset = new Supervised($samples, $outcomes);
    }

    public function test_build_supervised_dataset()
    {
        $this->assertInstanceOf(Supervised::class, $this->dataset);
    }

    public function test_get_outcomes()
    {
        $this->assertEquals(['not monster', 'monster', 'not monster', 'monster'], $this->dataset->outcomes());
    }

    public function test_randomize()
    {
        $this->dataset->randomize();

        $this->assertTrue(true);
    }

    public function test_head()
    {
        $this->assertEquals(3, $this->dataset->head(3)->rows());
    }

    public function test_split_dataset()
    {
        $splits = $this->dataset->split(0.5);

        $this->assertEquals(2, count($splits[0]));
        $this->assertEquals(2, count($splits[1]));
    }

    public function test_take_samples_from_dataset()
    {
        $this->assertEquals(4, $this->dataset->count());

        $dataset = $this->dataset->take(3);

        $this->assertEquals(3, $dataset->count());
        $this->assertEquals(1, $this->dataset->count());
    }

    public function test_leave_samples_in_dataset()
    {
        $this->assertEquals(4, $this->dataset->count());

        $dataset = $this->dataset->leave(1);

        $this->assertEquals(3, $dataset->count());
        $this->assertEquals(1, $this->dataset->count());
    }

    public function test_generate_random_weighted_subset_with_replacement()
    {
        $this->dataset->setWeight(3, 0);
        $this->dataset->setWeight(0, 0);
        $this->dataset->setWeight(1, 0);
        $this->dataset->setWeight(2, 1);

        $samples = $this->dataset->generateRandomWeightedSubsetWithReplacement(0.25)->all();

        $this->assertEquals(['nice', 'rough', 'friendly', 'not monster'], $samples[0]);
    }
}
