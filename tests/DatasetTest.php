<?php

use Rubix\Engine\Dataset;
use PHPUnit\Framework\TestCase;

class DatasetTest extends TestCase
{
    protected $data;

    public function setUp()
    {
        $data = [
            ['nice', 'furry', 'friendly', 'not monster'],
            ['mean', 'furry', 'loner', 'monster'],
            ['nice', 'rough', 'friendly', 'not monster'],
            ['mean', 'rough', 'friendly', 'monster'],
        ];

        $this->dataset = new Dataset($data);
    }

    public function test_build_supervised_dataset()
    {
        $this->assertInstanceOf(Dataset::class, $this->dataset);
    }

    public function test_randomize()
    {
        $this->dataset->randomize();

        $this->assertTrue(true);
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
}
