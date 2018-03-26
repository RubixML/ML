<?php

use Rubix\Engine\SupervisedDataset;
use PHPUnit\Framework\TestCase;

class SupervisedDatasetTest extends TestCase
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

        $this->data = new SupervisedDataset($data);
    }

    public function test_build_supervised_dataset()
    {
        $this->assertInstanceOf(SupervisedDataset::class, $this->data);
    }

    public function test_get_outcomes()
    {
        $this->assertEquals(['not monster', 'monster', 'not monster', 'monster'], $this->data->outcomes());
    }

    public function test_randomize()
    {
        $this->data->randomize();

        $this->assertTrue(true);
    }

    public function test_split_dataset()
    {
        $splits = $this->data->split(0.5);

        $this->assertEquals(2, count($splits[0]));
        $this->assertEquals(2, count($splits[1]));
    }
}
