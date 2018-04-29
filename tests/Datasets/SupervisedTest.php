<?php

use Rubix\Engine\Datasets\Dataset;
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
        $this->assertInstanceOf(Dataset::class, $this->dataset);
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
        $this->assertEquals(3, $this->dataset->head(3)->count());
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

    public function test_split_dataset()
    {
        $splits = $this->dataset->split(0.5);

        $this->assertEquals(2, count($splits[0]));
        $this->assertEquals(2, count($splits[1]));
    }

    public function test_stratified_split()
    {
        $splits = $this->dataset->stratifiedSplit(0.5);

        $this->assertEquals(2, count($splits[0]));
        $this->assertEquals(2, count($splits[1]));
    }

    public function test_fold_dataset()
    {
        $folds = $this->dataset->fold(1);

        $this->assertEquals(2, count($folds));
        $this->assertEquals(2, count($folds[0]));
        $this->assertEquals(2, count($folds[1]));
    }

    public function test_stratified_fold()
    {
        $folds = $this->dataset->stratifiedFold(1);

        $this->assertEquals(2, count($folds));
        $this->assertEquals(2, count($folds[0]));
        $this->assertEquals(2, count($folds[1]));
    }
}
