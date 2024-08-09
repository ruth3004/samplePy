% Specify the file path
file_path = '20220427_RM0008_126hpf_fP3_f3_fluorescence_data.h5';

% Read the data
raw_traces = h5read(file_path, '/20220427_RM0008_126hpf_fP3_f3/raw_traces')'; %transpose is important!
labels = h5read(file_path, '/20220427_RM0008_126hpf_fP3_f3/label_lm_plane');
planes = h5read(file_path, '/20220427_RM0008_126hpf_fP3_f3/plane_nr');
trials = h5read(file_path, '/20220427_RM0008_126hpf_fP3_f3/trial_nr');
odors = h5read(file_path, '/20220427_RM0008_126hpf_fP3_f3/odor');

% Plot the first three traces (as in your images)
figure;
hold on;

% Assuming raw_traces is a 2D array where each row is a trace
% and planes==7 selects the traces you want to plot
selected_traces = raw_traces(planes==7,:);

% Plot only the first three traces
for i = 1:min(3, size(selected_traces, 1))
    plot(linspace(0, 2, size(selected_traces, 2)), selected_traces(i, :), 'LineWidth', 2);
end

% Set axis limits and labels
xlabel('Time (frames)');
ylabel('Fluorescence Intensity');

% Add title and legend
title('Fluorescence Intensity Traces');
legend('Label 1', 'Label 2', 'Label 3');

% Customize the plot appearance
grid on;
set(gca, 'FontSize', 12);

hold off;
