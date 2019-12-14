<?php

namespace Rubix\ML\Tests\Datasets\Extractors;

use Rubix\ML\Datasets\Labeled;
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
        $this->factory = new NDJSON('tests/test.ndjson');
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
            ['nice', 'furry', 'friendly', 4, 'not monster'],
            ['mean', 'furry', 'loner', -1.5, 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5, 'not monster'],
        ];

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
    }

    public function test_extract_with_labels() : void
    {
        $dataset = $this->factory->extractWithLabels();

        $samples = [
            ['nice', 'furry', 'friendly', 4],
            ['mean', 'furry', 'loner', -1.5],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1],
            ['nice', 'rough', 'friendly', 2.9],
            ['nice', 'furry', 'loner', -5],
        ];

        $labels = [
            'not monster', 'monster', 'not monster', 'monster',
            'not monster', 'not monster',
        ];

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
        $this->assertEquals($labels, $dataset->labels());
    }
}
