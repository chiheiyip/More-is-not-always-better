function output_csv = export_eeg_set_metadata(eeg_root, output_csv)
%EXPORT_EEG_SET_METADATA Export lightweight .set provenance without loading .fdt.

files = dir(fullfile(char(eeg_root), '*.set'));
if isempty(files)
    error('No .set files found in %s', char(eeg_root));
end

rows = table();
for i = 1:numel(files)
    set_path = fullfile(files(i).folder, files(i).name);
    E = load(set_path, '-mat', 'history', 'pnts', 'srate', 'nbchan', 'event', 'datfile', 'chanlocs');
    [~, participant_id] = fileparts(files(i).name);
    history_easy_path = "";
    if isfield(E, 'history')
        token = regexp(E.history, 'pop_easy\(''([^'']+)''', 'tokens', 'once');
        if ~isempty(token)
            history_easy_path = string(token{1});
        end
    end
    types = strings(1, numel(E.event));
    latencies = strings(1, numel(E.event));
    for j = 1:numel(E.event)
        types(j) = string(E.event(j).type);
        latencies(j) = string(round(double(E.event(j).latency)));
    end
    labels = strings(1, numel(E.chanlocs));
    for j = 1:numel(E.chanlocs)
        labels(j) = string(E.chanlocs(j).labels);
    end
    row = table( ...
        string(participant_id), string(set_path), history_easy_path, ...
        double(E.srate), double(E.pnts), double(E.nbchan), string(E.datfile), ...
        strjoin(types, '|'), strjoin(latencies, '|'), strjoin(labels, '|'), ...
        'VariableNames', {'participant_id','set_path','history_easy_path','srate_hz','n_samples','n_channels','datfile','event_types','event_latencies','channel_labels'});
    rows = [rows; row]; %#ok<AGROW>
end

output_csv = char(output_csv);
out_parent = fileparts(output_csv);
if ~isempty(out_parent) && exist(out_parent, 'dir') ~= 7
    mkdir(out_parent);
end
writetable(rows, output_csv, 'Encoding', 'UTF-8');
fprintf('Wrote %s\n', output_csv);
end
