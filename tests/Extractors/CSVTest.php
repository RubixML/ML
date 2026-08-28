<?php

namespace Rubix\ML\Tests\Extractors;

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\Exporter;
use Rubix\ML\Extractors\Extractor;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;
use Traversable;

use function sys_get_temp_dir;
use function tempnam;
use function file_put_contents;
use function unlink;

/**
 * @group Extractors
 * @covers \Rubix\ML\Extractors\CSV
 */
class CSVTest extends TestCase
{
    /**
     * @var \Rubix\ML\Extractors\CSV;
     */
    protected $extractor;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->extractor = new CSV('tests/test.csv', true, ',', '"');
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(CSV::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
        $this->assertInstanceOf(Exporter::class, $this->extractor);
        $this->assertInstanceOf(IteratorAggregate::class, $this->extractor);
        $this->assertInstanceOf(Traversable::class, $this->extractor);
    }

    /**
     * @test
     */
    public function header() : void
    {
        $expected = [
            'attitude', 'texture', 'sociability', 'rating', 'class',
        ];

        $this->assertEquals($expected, $this->extractor->header());
    }

    /**
     * @test
     */
    public function extractExport() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => '4', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-1.5', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.6', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '-1', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.9', 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-5', 'class' => 'not monster'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, $records);

        $expected = [
            'attitude', 'texture', 'sociability', 'rating', 'class',
        ];

        $header = $this->extractor->header();

        $this->assertEquals($expected, $header);

        $this->extractor->export($records);

        $this->assertFileExists('tests/test.csv');
    }

    /**
     * @test
     */
    public function extractSkipsBlankLines() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'csv_');

        file_put_contents($path, "attitude,texture\nnice,furry\n\nmean,rough\n\n\n");

        $extractor = new CSV($path, true, ',', '"');

        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry'],
            ['attitude' => 'mean', 'texture' => 'rough'],
        ];

        $this->assertEquals($expected, iterator_to_array($extractor, false));

        unlink($path);
    }

    /**
     * @test
     */
    public function extractMalformedRecord() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'csv_');

        file_put_contents($path, "attitude,texture,sociability\nnice,furry\n");

        $extractor = new CSV($path, true, ',', '"');

        $this->expectException(RuntimeException::class);

        iterator_to_array($extractor, false);

        unlink($path);
    }
}
