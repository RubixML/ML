<?php

namespace Rubix\ML\Tests\Other\Helpers;

use PHPUnit\Framework\TestCase;
use Rubix\ML\Other\Helpers\JSON;

/**
 * @group Helpers
 * @covers \Rubix\ML\Other\Helpers\JSON
 */
class JSONTest extends TestCase
{
    /**
     * @test
     */
    public function decodeSuccess() : void
    {
        $data = (string) file_get_contents('tests/test.json');
        $actual = JSON::decode($data);

        $expect = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => 4, 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => -1.5, 'class' => 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5, 'not monster'],
        ];

        $this->assertSame($expect, $actual);
    }

    /**
     * @test
     */
    public function decodeFail() : void
    {
        $this->expectException(\RuntimeException::class);
        $invalid = '[{"package":...}]';
        JSON::decode($invalid);
    }

    /**
     * @test
     */
    public function encode() : void
    {
        $actual = JSON::encode(['package' => 'rubix/ml']);
        $expect = '{"package":"rubix\/ml"}';
        $this->assertSame($expect, $actual);
    }
}
