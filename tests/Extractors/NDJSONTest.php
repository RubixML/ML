<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Extractors\NDJSON;
use PHPUnit\Framework\TestCase;

use function sys_get_temp_dir;
use function file_put_contents;
use function file_get_contents;
use function uniqid;
use function unlink;

#[Group('Extractors')]
#[CoversClass(NDJSON::class)]
class NDJSONTest extends TestCase
{
    protected NDJSON $extractor;

    protected function setUp() : void
    {
        $this->extractor = new NDJSON('tests/test.jsonl');
    }

    #[Test]
    public function extractExport() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => 4.0, 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => -1.5, 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => 2.6, 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => -1.0, 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => 2.9, 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => -5.0, 'class' => 'not monster'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, array_values($records));

        $this->extractor->export($records, true);

        $this->assertFileExists('tests/test.jsonl');
    }

    #[Test]
    public function exportAppendNoTrailingNewline() : void
    {
        $path = sys_get_temp_dir() . '/ndjson_' . uniqid() . '.ndjson';

        file_put_contents($path, '{"attitude":"nice","rating":4.0}');

        $extractor = new NDJSON($path);

        $extractor->export([
            ['attitude' => 'shy', 'rating' => 2.5],
        ]);

        $extractor->export([
            ['attitude' => 'mean', 'rating' => -1.0],
        ]);

        $this->assertSame('{"attitude":"nice","rating":4.0}' . PHP_EOL
            . '{"attitude":"shy","rating":2.5}' . PHP_EOL
            . '{"attitude":"mean","rating":-1.0}' . PHP_EOL, file_get_contents($path));

        $this->assertEquals([
            ['attitude' => 'nice', 'rating' => 4.0],
            ['attitude' => 'shy', 'rating' => 2.5],
            ['attitude' => 'mean', 'rating' => -1.0],
        ], iterator_to_array($extractor, false));

        unlink($path);
    }

    #[Test]
    public function exportOverwriteNoTrailingNewline() : void
    {
        $path = sys_get_temp_dir() . '/ndjson_' . uniqid() . '.ndjson';

        file_put_contents($path, '{"attitude":"nice","rating":4.0}');

        $extractor = new NDJSON($path);

        $extractor->export([
            ['attitude' => 'shy', 'rating' => 2.5],
        ], overwrite: true);

        $this->assertSame('{"attitude":"shy","rating":2.5}' . PHP_EOL, file_get_contents($path));

        unlink($path);
    }
}
