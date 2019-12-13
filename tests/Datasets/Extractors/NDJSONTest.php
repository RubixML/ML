<?php

namespace Rubix\ML\Tests\Datasets\Extractors;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Extractors\NDJSON;
use Rubix\ML\Datasets\Extractors\Extractor;
use PHPUnit\Framework\TestCase;

class NDJSONTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Extractors\NDJSON;
     */
    protected $factory;

    public function setUp() : void
    {
        $this->factory = new NDJSON('tests/unlabeled.ndjson');
    }

    public function test_build_factory() : void
    {
        $this->assertInstanceOf(NDJSON::class, $this->factory);
        $this->assertInstanceOf(Extractor::class, $this->factory);
    }

    public function test_extract() : void
    {
        $dataset = $this->factory->extract();

        $samples = [
            ['nice', 'furry', 'friendly', '4'],
            ['mean', 'furry', 'loner', '-1.5'],
            ['nice', 'rough', 'friendly', '2.6'],
            ['mean', 'rough', 'friendly', '-1'],
            ['nice', 'rough', 'friendly', '2.9'],
            ['nice', 'furry', 'loner', '-5'],
        ];

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
    }
}
