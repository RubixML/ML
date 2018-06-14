<?php

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use PHPUnit\Framework\TestCase;

class LabeledTest extends TestCase
{
    protected $dataset;

    protected $samples;

    protected $labels;

    public function setUp()
    {
        $this->samples = [
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['nice', 'furry', 'loner'],
        ];

        $this->labels = [
            'not monster', 'monster', 'not monster',
            'monster', 'not monster', 'not monster'];

        $this->dataset = new Labeled($this->samples, $this->labels);
    }

    public function test_build_supervised_dataset()
    {
        $this->assertInstanceOf(Labeled::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
    }

    public function test_get_labels()
    {
        $this->assertEquals($this->labels, $this->dataset->labels());
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

    public function test_tail()
    {
        $this->assertEquals(3, $this->dataset->tail(3)->count());
    }

    public function test_take_samples_from_dataset()
    {
        $this->assertEquals(6, $this->dataset->count());

        $dataset = $this->dataset->take(3);

        $this->assertEquals(3, $dataset->count());
        $this->assertEquals(3, $this->dataset->count());
    }

    public function test_leave_samples_in_dataset()
    {
        $this->assertEquals(6, $this->dataset->count());

        $dataset = $this->dataset->leave(1);

        $this->assertEquals(5, $dataset->count());
        $this->assertEquals(1, $this->dataset->count());
    }

    public function test_split_dataset()
    {
        list($left, $right) = $this->dataset->split(0.5);

        $this->assertEquals(3, count($left));
        $this->assertEquals(3, count($right));
    }

    public function test_stratified_split()
    {
        list($left, $right) = $this->dataset->split(0.5);

        $this->assertEquals(3, count($left));
        $this->assertEquals(3, count($right));
    }

    public function test_fold_dataset()
    {
        $folds = $this->dataset->fold(2);

        $this->assertEquals(2, count($folds));
        $this->assertEquals(3, count($folds[0]));
        $this->assertEquals(3, count($folds[1]));
    }

    public function test_stratified_fold()
    {
        $folds = $this->dataset->stratifiedFold(2);

        $this->assertEquals(2, count($folds));
        $this->assertEquals(3, count($folds[0]));
        $this->assertEquals(3, count($folds[1]));
    }
}
