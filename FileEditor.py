from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def remove_silence(input_file='input.aif', output_file='out.aif', silence_thresh=-40, min_silence_len=300):
    '''
    Remove silence from an audio file
    :param input_file: path to the input audio file
    :param output_file:
    :param silence_thresh:
    :param min_silence_len:
    :return:
    '''

    # Loading audio file
    print(f'Removing silence from {input_file} to {output_file}')
    sound = AudioSegment.from_file(input_file, format="aif")
    print(f'Loaded sound: {sound}\n Duration: {len(sound)/ 1000} seconds \n Frame rate: {sound.frame_rate}')
    print(f'\nChannels: {sound.channels}\n Frame width: {sound.frame_width}')

    # Splitting audio file
    # a list of AudioSegment objects, for example: [ <AudioSegment1>, <AudioSegment2>, ... ]
    # where: <AudioSegment1> = sound[start1:end1], <AudioSegment2> = sound[start2:end2], ...
    chunks = split_on_silence(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    print(f'Number of chunks: {len(chunks)}')
    print(f'Chunks: {chunks}')
    print(f'Chunks[0]: {chunks[0]}')

    non_silence_duration = sum([len(chunk) for chunk in chunks])
    print(f'None silence duration: {non_silence_duration/1000} seconds')
    print(f'Silence duration: {len(sound)/1000 - non_silence_duration/1000} seconds')

    # we will keep silence for a few seconds between each segment
    keep_silence = 150  # in milliseconds
    out = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        out += chunk
        if i < len(chunks) - 1:
            out += AudioSegment.silent(duration=keep_silence)
    # Exporting audio file
    # Avoid overwriting existing files
    while os.path.exists(output_file):
        output_file = output_file.replace('.aif', '_new.aif')
    try:
        out.export(output_file, format="aif")
        print(f'Exported {output_file}')
        return output_file
    except Exception as e:
        print(f'Error: {e}')
        print(f'Failed to export {output_file}')
        output_file = None
        return


if __name__ == '__main__':
    input_file = 'files/input.aif'
    remove_silence(input_file)